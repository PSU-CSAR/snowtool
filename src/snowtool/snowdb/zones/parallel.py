"""An ordered parallel-map / serial-reduce engine for streaming block work.

This is the concurrency core lifted out of the terrain generator (and reusable by
any streaming generator with the same shape): a pure, expensive per-item ``compute``
runs on a worker pool while ``reduce`` folds each result serially, in item order, on
the calling thread. The design is dominated by five load-bearing invariants -- read
:func:`ordered_parallel_map` and its docstrings before touching this, because the
engine guards against real crashes (a leaked worker inside a GDAL dataset the caller
closes the instant the engine returns is a use-after-free / SIGSEGV):

1. ``reduce`` is called serially, in item order, on the calling thread -- so float
   accumulation is bit-for-bit reproducible (parallel == serial).
2. No worker may still be executing ``compute`` when the engine returns, under any
   number of ctrl+c presses.
3. Cancellation makes queued computes no-op: a worker observes the cancel (via its
   :class:`CancelToken`) before doing expensive work, so an aborting run drains at
   most the one item already in flight, not every queued item's (possibly network)
   read.
4. The warm-gate confines thread starts to a phase where a leaked thread can never be
   inside real work.
5. A swallowed interrupt re-raises on clean exit of the teardown loop (unless one is
   already propagating).
"""

from __future__ import annotations

import os
import sys
import threading

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

from snowtool.snowdb.progress import NULL_PROGRESS, ProgressReporter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# Default cap on the *auto* worker count (``workers=None``): one thread per CPU but
# never more than this. Beyond ~here the per-item work stops scaling (reads are
# serialised under a lock and the serial reduce starts to bind) while memory and
# lock contention keep climbing, so more threads mostly cost RAM. An explicit
# ``workers`` is always honoured -- the caller owns that tradeoff.
MAX_AUTO_WORKERS = 16


def effective_workers(requested: int | None) -> int:
    """Resolve the worker count.

    ``requested`` of ``None`` means auto -- one thread per CPU, but never more than
    :data:`MAX_AUTO_WORKERS`. ``1`` (or anything <= 1) means the serial path. An
    explicit request is honoured as-is: the caller owns the memory tradeoff (bound
    per-worker RAM with a smaller block/item; see the terrain module docstring).
    Always >= 1.
    """
    if requested is None:
        return min(os.cpu_count() or 1, MAX_AUTO_WORKERS)
    return max(1, requested)


class CancelToken:
    """The engine's cancellation signal, handed to every ``compute`` call.

    The engine owns the underlying :class:`threading.Event` and is the only thing
    that sets it (in teardown). A ``compute`` callable reads :attr:`cancelled` to
    short-circuit expensive work once a run is aborting -- typically both before it
    queues on any shared read lock and again after acquiring it, to close the race
    for a worker that passed the first check just before cancellation.
    """

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    @property
    def cancelled(self) -> bool:
        """True once the engine is tearing down (e.g. a ctrl+c is propagating)."""
        return self._event.is_set()


def ordered_parallel_map[T, R](
    items: Iterable[T],
    compute: Callable[[T, CancelToken], R | None],
    reduce: Callable[[R], None],
    *,
    workers: int,
    progress: ProgressReporter = NULL_PROGRESS,
    label: str = 'processing',
) -> None:
    """Map ``compute`` over ``items`` in parallel, folding results serially in order.

    ``compute(item, token)`` is pure and runs on a worker pool; it may return the
    skip sentinel ``None`` to contribute nothing. ``reduce(result)`` runs on the
    calling thread, once per non-``None`` result, in ``items`` order -- so any float
    accumulation it does is bit-for-bit independent of ``workers`` (invariant 1).
    ``workers <= 1`` forces the serial path; otherwise a sliding window of at most
    ``workers`` in-flight futures keeps every worker fed. ``progress`` gets one
    tracked task labelled ``label``, advanced once per item.

    On any exception (including ctrl+c) the pool is torn down by hand so no worker is
    left running ``compute`` when this returns (invariants 2-5); see
    :func:`_teardown`.
    """
    blocks = list(items)
    # One bar over the whole pass; the serial, ordered reduce is the unit of visible
    # progress (advance once per item, computed serial or parallel).
    with progress.track(label, total=len(blocks)) as task:
        # The engine owns the cancellation Event and is the sole setter (in teardown);
        # compute observes it through the token (invariant 3).
        cancelled = threading.Event()
        token = CancelToken(cancelled)

        if workers <= 1:
            for block in blocks:
                result = compute(block, token)
                if result is not None:
                    reduce(result)
                task.advance()
            return

        # Parallel map, serial ordered reduce. A sliding window of at most
        # ``workers`` in-flight futures keeps every worker fed while bounding the
        # transient memory of buffered results; popping left to right reduces in
        # item order, so the accumulation stays deterministic (invariant 1).
        # The pool is torn down by hand rather than as a context manager: on an
        # interrupt, ``__exit__``'s plain ``shutdown(wait=True)`` would drain every
        # queued item's read, and a second ctrl+c could abort the join entirely --
        # leaving a worker inside the shared resource when the caller closes it (a
        # use-after-free in GDAL).
        pending = iter(blocks)
        pool = ThreadPoolExecutor(max_workers=workers)
        window: deque[Future[R | None]] = deque()
        # Opened by the teardown; warm-up tasks (below) block on it so that every
        # worker thread exists before any item work is queued.
        warm_gate = threading.Event()
        try:
            # Pre-start the full complement of worker threads on gated no-ops. A
            # ctrl+c inside Thread.start can leak an OS thread the executor never
            # registered (CPython registers a worker *after* starting it), and
            # shutdown() never joins an unregistered thread -- confining thread
            # starts to this warm-up, when only harmless gate-waits are queued,
            # means a leaked thread can never be the one inside real work
            # (invariant 4). Afterwards the pool is at max_workers, so the submits
            # below never start a thread.
            for _ in range(workers):
                pool.submit(warm_gate.wait)
            warm_gate.set()
            for block in pending:
                window.append(pool.submit(compute, block, token))
                if len(window) >= workers:
                    break
            while window:
                result = window.popleft().result()
                if result is not None:
                    reduce(result)
                task.advance()
                next_block = next(pending, None)
                if next_block is not None:
                    window.append(pool.submit(compute, next_block, token))
        finally:
            _teardown(pool, window, warm_gate, cancelled)


def _teardown[R](
    pool: ThreadPoolExecutor,
    window: deque[Future[R | None]],
    warm_gate: threading.Event,
    cancelled: threading.Event,
) -> None:
    """Join every worker, surviving any number of further ctrl+c presses.

    The caller may free a shared resource (e.g. close a WarpedVRT and its GDAL
    source datasets) as soon as :func:`ordered_parallel_map` unwinds, so no worker
    may still be running ``compute`` when this returns -- this join is what stands
    between a ctrl+c and a use-after-free (SIGSEGV) inside GDAL (invariant 2), and
    further ctrl+c presses must not be able to skip it: the loop retries until the
    join completes uninterrupted. It catches BaseException, not KeyboardInterrupt,
    because an interrupt landing inside the wait/join internals can surface as a
    secondary error (e.g. a RuntimeError from a broken Condition) that must not
    abort the join either.

    Cancelling first makes still-queued workers no-op instead of doing expensive
    work, so the join waits on at most the one item already in flight (invariant 3).
    On a clean exit nothing is in flight and the flag is inert.
    """
    cancelled.set()
    warm_gate.set()  # unblock warm-up no-ops so shutdown can join
    interrupted = False
    while True:
        try:
            wait(window)
            pool.shutdown(wait=True, cancel_futures=True)
        except BaseException:  # noqa: BLE001 -- see the docstring
            interrupted = True
            continue
        break
    # An interrupt swallowed by the retry loop must still surface -- unless one is
    # already propagating out of the caller's try (raising here would replace it
    # with a bare KeyboardInterrupt for nothing) (invariant 5).
    if interrupted and sys.exc_info()[1] is None:
        raise KeyboardInterrupt
