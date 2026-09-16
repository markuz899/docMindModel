# Troubleshooting

## The recipient did not receive the document

Check the request status first with `GET /documents/{id}`. If the status is
`rendered` but there is no delivery row, the delivery step was never invoked.
If deliveries exist, `GET /documents/{id}/deliveries` shows the provider, the
status and the `errorCode` of each attempt.

## A provider was never called

`ProviderRouter` rejects a delivery before calling any provider when the channel
has no mapping in `docsync.channels`. In that case the delivery is recorded with
`unsupported_channel` and no `provider_message_id` is present.

## Documents are produced but not signed

`documents.signed` is false only if the signing step was skipped, which happens
when the request was replayed from a pre-signature checkpoint. The HSM
certificate itself cannot be missing at runtime, because the application fails
to start without it.
