# Mesh Gateway

## Purpose

Mesh Gateway is the single ingress for every public API call at Acme. It is a Go
service that terminates TLS, authenticates the caller, applies rate limits and
forwards the request to the right upstream service. It holds no business logic
and no database.

## Position in the platform

Clients talk only to the gateway. Internal services are not reachable from
outside the cluster. The gateway adds the `X-Request-Id` and `X-Caller-Id`
headers that downstream services log and propagate.
