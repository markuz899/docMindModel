# Configuration

## Property sources

Configuration lives in `application.yaml` and is overridden per environment by
`application-{profile}.yaml` and by environment variables. Spring relaxed
binding applies: `docsync.render.timeout-ms` can be set as
`DOCSYNC_RENDER_TIMEOUT_MS`.

## Render settings

`docsync.render.timeout-ms` defaults to 20000 and bounds a single render.
`docsync.render.max-payload-kb` defaults to 512. Exceeding it rejects the
request with `payload_too_large`.

## Channel routing

`docsync.channels` maps a channel name to a provider bean name, for example
`email: emailDeliveryProvider`. A channel absent from this map is unsupported at
runtime.

## Storage settings

`docsync.storage.bucket` is required. `docsync.storage.download-url-ttl-seconds`
defaults to 900 and controls the lifetime of the download URL returned by
`GET /documents/{id}`.
