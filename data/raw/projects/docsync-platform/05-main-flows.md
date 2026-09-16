# Main Flows

## Document production flow

1. `POST /documents` stores a `document_requests` row with status `received`.
2. The request is queued; a worker moves it to `rendering`.
3. `RenderService` resolves the active template and renders the PDF.
4. `SignatureService` seals the PDF.
5. `StorageService` uploads it and writes the `documents` row.
6. The request moves to `rendered`.

## Delivery flow

1. Once the request is `rendered`, `DeliveryService.deliver` is invoked.
2. `ProviderRouter` resolves the channel to a provider bean.
3. The provider is called; a `deliveries` row is written with status `queued`
   before the call and updated to `sent` when the provider accepts it.
4. The provider later calls `POST /providers/callbacks/{provider}`.
5. `DeliveryStatusListener` updates the delivery to `accepted`, `rejected` or
   `bounced`.

## Retry flow

A delivery in `rejected` state is retried only if the provider reported a
retryable error code. Retries create a new `deliveries` row with an incremented
`attempt_number`. A `bounced` delivery is never retried.
