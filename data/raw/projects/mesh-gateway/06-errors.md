# Errors

## Gateway errors

`route_not_found` (HTTP 404) when no route matches. `unauthenticated` (HTTP 401)
when the bearer token is missing or invalid. `forbidden_scope` (HTTP 403) when
the route requires a scope the caller does not have. `rate_limited` (HTTP 429)
when the token bucket is empty.

## Upstream errors

`upstream_timeout` (HTTP 504) when the upstream exceeds the route timeout.
`upstream_unavailable` (HTTP 502) when the connection fails. The gateway never
converts an upstream 5xx into a 2xx: upstream status codes above 499 are passed
through unchanged, with the body replaced by the gateway error envelope.

## Error envelope

`{ "error": "<code>", "requestId": "<id>" }`. The `requestId` matches the
`X-Request-Id` header and is the key to correlate gateway and upstream logs.
