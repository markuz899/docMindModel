# Errors

## Request errors

`payload_too_large` (HTTP 413) when the payload exceeds
`docsync.render.max-payload-kb`. `unknown_template` (HTTP 404) when
`templateCode` has no active row in `templates`. `unsupported_channel`
(HTTP 400) when `ProviderRouter` has no mapping for the channel.

## Render errors

`render_timeout` when the render exceeds `docsync.render.timeout-ms`.
`template_render_error` when the template engine throws. Both move the request
to `render_failed`; neither is retried automatically.

## Signature errors

`signature_unavailable` when the HSM is unreachable at signing time. The request
stays in `rendering` and the job is retried by the worker.

## Delivery errors

Provider errors are normalised to `provider_rejected`, `provider_unavailable`
and `recipient_invalid`. `recipient_invalid` is final. `provider_unavailable` is
retryable. `provider_rejected` is retryable only when the provider marks the
error code as retryable.
