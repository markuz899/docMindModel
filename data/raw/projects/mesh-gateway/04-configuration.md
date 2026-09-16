# Configuration

## Files and env

`gateway.yaml` holds static settings; `routes.yaml` holds the route table.
Environment variables prefixed with `GW_` override any key, for example
`GW_UPSTREAM_TIMEOUT_MS`.

## Timeouts

`upstreamTimeoutMs` is the default upstream timeout and is 30000. A route can
override it with its own `timeoutMs`. `readHeaderTimeoutMs` is 5000 and
`idleTimeoutMs` is 120000.

## Rate limiting

`rateLimit.defaultRps` is 50 requests per second per caller.
`rateLimit.burst` is 100. `rateLimit.redisUrl` is required; without it the
gateway refuses to start.

## TLS

`tls.certPath` and `tls.keyPath` are required in production profiles. `tls.minVersion`
defaults to `1.2`.
