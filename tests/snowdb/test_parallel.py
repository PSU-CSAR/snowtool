"""The ordered parallel-map engine on plain data with real threads.

These pin the engine's invariants directly, without the terrain machinery: the
reduce sees results in item order regardless of worker count, ``None`` results are
skipped, the serial path (workers <= 1) is taken, and teardown drains every worker
before returning even when an exception is raised mid-reduce. Synchronisation is by
:class:`threading.Barrier`/:class:`threading.Event`, never ``sleep``, so the tests
are deterministic.
"""

import os
import threading

import pytest

from snowtool.snowdb.zones.parallel import (
    MAX_AUTO_WORKERS,
    CancelToken,
    effective_workers,
    ordered_parallel_map,
)


def test_effective_workers_defaults_and_honors_explicit_request():
    # An explicit request is honoured as-is -- no silent memory override.
    assert effective_workers(4) == 4
    assert effective_workers(64) == 64
    # <= 1 is the serial path.
    assert effective_workers(1) == 1
    assert effective_workers(0) == 1
    # None means auto: one thread per CPU, but never more than the cap.
    auto = effective_workers(None)
    assert auto == min(os.cpu_count() or 1, MAX_AUTO_WORKERS)
    assert 1 <= auto <= MAX_AUTO_WORKERS


@pytest.mark.parametrize('workers', [2, 4, 8])
def test_reduce_runs_in_item_order_under_parallel_compute(workers):
    # A barrier forces the first `workers` computes to run concurrently (none can
    # return until all have started), so completion order is genuinely up to the
    # scheduler. reduce must still see strict item order -- that ordering is what
    # makes float accumulation reproducible.
    items = list(range(64))
    barrier = threading.Barrier(min(workers, len(items)))

    def compute(item, cancel):
        # Only synchronise the first `workers` items so the barrier can't deadlock
        # on the trailing window; the point is that several run concurrently.
        if item < workers:
            barrier.wait()
        return item

    reduced = []
    ordered_parallel_map(items, compute, reduced.append, workers=workers)
    assert reduced == items


def test_serial_path_runs_on_calling_thread():
    # workers=1 takes the serial branch: every compute AND reduce runs on the
    # calling thread (no pool), in item order.
    main = threading.get_ident()
    items = list(range(10))
    compute_threads = []
    reduce_threads = []

    def compute(item, cancel):
        compute_threads.append(threading.get_ident())
        return item

    def reduce(result):
        reduce_threads.append(threading.get_ident())

    ordered_parallel_map(items, compute, reduce, workers=1)
    assert all(t == main for t in compute_threads)
    assert all(t == main for t in reduce_threads)
    assert len(compute_threads) == len(items)


@pytest.mark.parametrize('workers', [1, 4])
def test_none_results_are_not_reduced(workers):
    # compute returning the skip sentinel None contributes nothing to reduce.
    items = list(range(20))

    def compute(item, cancel):
        return item if item % 2 == 0 else None

    reduced = []
    ordered_parallel_map(items, compute, reduced.append, workers=workers)
    assert reduced == [i for i in items if i % 2 == 0]


def test_compute_receives_a_cancel_token_not_set_on_clean_run():
    # On a clean pass the token is never set -- computes see cancelled == False.
    seen = []

    def compute(item, cancel):
        assert isinstance(cancel, CancelToken)
        seen.append(cancel.cancelled)
        return item

    ordered_parallel_map([1, 2, 3], compute, lambda r: None, workers=4)
    assert seen == [False, False, False]


def test_teardown_drains_workers_after_exception_mid_reduce():
    # An exception raised inside reduce must still drain every in-flight worker
    # before ordered_parallel_map returns: no compute may be executing when it
    # unwinds (the terrain caller closes the WarpedVRT the instant it does). We
    # track live computes with a lock and assert it settled to zero.
    workers = 4
    live_lock = threading.Lock()
    live = 0
    max_live = 0
    # Rendezvous of the reducer with the other workers-1 in-flight computes:
    # reduce raises only once all of them are provably inside compute, and those
    # computes exit only when teardown signals cancellation -- so the drain is
    # genuinely exercised, with no timing assumptions.
    inflight = threading.Barrier(workers)
    idle = threading.Event()  # never set; a waitable to poll the token with

    def compute(item, cancel):
        nonlocal live, max_live
        with live_lock:
            live += 1
            max_live = max(max_live, live)
        try:
            if item == 0:
                # The first item completes immediately so reduce gets a result
                # while every other worker is still busy.
                return item
            inflight.wait()
            # Hold the worker inside compute until the engine's teardown cancels;
            # only cancellation can release it (reduce has already raised).
            while not cancel.cancelled:
                idle.wait(timeout=0.01)
            return item
        finally:
            with live_lock:
                live -= 1

    def reduce(result):
        # Blow up on the very first result, with every other worker mid-compute.
        inflight.wait()
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        ordered_parallel_map(range(32), compute, reduce, workers=workers)

    # Teardown joined every worker: none is still executing compute.
    with live_lock:
        assert live == 0
    # The barrier guaranteed workers-1 computes were live at once.
    assert max_live >= workers - 1
