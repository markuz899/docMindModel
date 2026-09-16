# Architecture

## Request pipeline

Each request passes through the pipeline in order: `TLSTerminator`,
`RequestIdMiddleware`, `AuthMiddleware`, `RateLimiter`, `RouteResolver`,
`UpstreamProxy`. A middleware that rejects a request short circuits the rest of
the pipeline.

## RouteResolver

`RouteResolver` matches the request path against the route table and produces an
upstream target. The route table is loaded from `routes.yaml` at startup and
refreshed on SIGHUP. An unmatched path returns HTTP 404 `route_not_found`
without contacting any upstream.

## RateLimiter

`RateLimiter` is a token bucket keyed by caller id and route. Buckets live in
Redis so all gateway replicas share the same budget. When the bucket is empty
the request is rejected with HTTP 429 and a `Retry-After` header.

## UpstreamProxy

`UpstreamProxy` performs the forwarded call. It applies the per-route timeout,
retries idempotent methods and records the upstream latency metric
`gateway_upstream_duration_seconds`.
