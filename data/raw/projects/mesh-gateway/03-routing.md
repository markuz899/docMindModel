# Routing

## routes.yaml

Each entry declares `path`, `methods`, `upstream`, `timeoutMs`, `authRequired`
and `rateLimit`. The first matching entry wins, so more specific paths must be
declared before wildcards.

## Identity routes

`/v1/me` and `/v1/auth/*` forward to the `acme-identity` upstream.
`/v1/me` requires authentication; `/v1/auth/login` does not.

## Billing routes

`/v1/invoices/*` and `/v1/subscriptions/*` forward to the `billing-core`
upstream and require the caller to carry a billing scope.

## Document routes

`/v1/documents/*` forwards to the `docsync-platform` upstream. Provider callback
paths under `/v1/providers/callbacks/*` are declared with `authRequired: false`
because providers authenticate with a signature instead of a bearer token.
