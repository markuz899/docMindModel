# API Reference

## POST /documents

Submits a document request. Body: `{ templateCode, payload, channel, recipient }`.
Returns HTTP 202 with `{ requestId, status: "received" }`. The work continues
asynchronously.

## GET /documents/{id}

Returns the document request with its current status and, once rendered, the
document id and a short-lived download URL.

## GET /documents/{id}/deliveries

Returns every delivery attempt for the document, newest first, including
`provider`, `status`, `attemptNumber` and `errorCode`.

## POST /providers/callbacks/{provider}

Endpoint used by delivery providers to report status changes. Handled by
`DeliveryStatusListener`. The payload is provider specific and normalised into
the `deliveries.status` values.
