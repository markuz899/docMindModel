# Architecture

## RenderService

`RenderService` resolves the template by `templateCode`, merges the payload and
produces a PDF through the `PdfRenderer` component. Rendering is synchronous and
bounded: a render that exceeds the render timeout is aborted and the request is
marked `render_failed`.

## SignatureService

`SignatureService` applies a PAdES seal using the certificate loaded from the
HSM at startup. If the certificate cannot be loaded the application refuses to
start. Signing happens after rendering and before storage.

## DeliveryService

`DeliveryService` selects a provider for each delivery and calls it through the
`DeliveryProvider` interface. Implementations are `EmailDeliveryProvider`,
`PecDeliveryProvider` and `PortalDeliveryProvider`. Selection is driven by the
`channel` field of the request, resolved by `ProviderRouter`.

## ProviderRouter

`ProviderRouter` maps a channel to a provider bean. If the channel has no
mapping the delivery is rejected immediately with `unsupported_channel` and no
provider is called. The routing table is built at startup from configuration.

## Storage

Rendered documents are stored in S3 under `documents/{tenantId}/{documentId}.pdf`.
`StorageService` is the only component that touches the S3 client. Object keys
are never reused; a re-render produces a new document id.
