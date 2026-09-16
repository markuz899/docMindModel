# Errors

## Domain errors

`invoice_not_draft` (HTTP 409) when issuing a non-draft invoice.
`invoice_not_payable` (HTTP 409) when charging an invoice that is not `open`.
`refund_exceeds_amount` (HTTP 422) when the refund amount is larger than the
paid amount.

## Provider errors

`StripeGateway` translates provider failures into `ProviderCardError`,
`ProviderRateLimited` and `ProviderUnavailable`. `ProviderCardError` is final:
the attempt is recorded and no retry is scheduled. `ProviderRateLimited` and
`ProviderUnavailable` are retryable and consume one of the
`PAYMENT_MAX_ATTEMPTS` attempts.

## Issuing failures

Issuing can fail before any charge: the invoice is not in `draft` state, the
number allocation conflicts (unique violation on `invoices.number`), or the PDF
render step fails and the job is retried by the worker. A failed issue leaves
the invoice in `draft`.

## Webhook failures

A webhook with an invalid signature is rejected with HTTP 400
`invalid_signature` and is not stored. A webhook whose `provider_event_id` is
already present is acknowledged with HTTP 200 and dropped.
