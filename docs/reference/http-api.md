# HTTP API reference

The read-only HTTP API, rendered from its OpenAPI schema. This is the same
contract served live at `/openapi.json` (and browsable at `/docs` and `/redoc`
on a running server) — here it is generated from the golden snapshot the test
suite pins against production, so it can't silently drift.

<div id="redoc-container"></div>
<script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
<script>
  Redoc.init('openapi.json', {}, document.getElementById('redoc-container'));
</script>
