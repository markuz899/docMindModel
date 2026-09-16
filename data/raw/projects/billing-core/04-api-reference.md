# API Reference

## POST /invoices

Creates a draft invoice for a subscription. Body: `{ subscriptionId, lines }`.
Returns HTTP 201 with the invoice in `draft` status. Creating an invoice does
not charge anything.

## POST /invoices/{id}/issue

Moves an invoice from `draft` to `open`, assigns the invoice `number`, renders
the PDF and enqueues the first payment attempt. Returns HTTP 202. Issuing is
rejected with HTTP 409 `invoice_not_draft` if the invoice is not in `draft`.

## POST /invoices/{id}/refund

Refunds a paid invoice through `PaymentService.refund`. Partial refunds are
accepted via an optional `amountCents`. Returns HTTP 200 with the refund record.

## GET /subscriptions/{id}

Returns the subscription with its current period and status. Requires the
`billing:read` scope.

## POST /webhooks/stripe

Receives Stripe callbacks. The route verifies the Stripe signature header
before doing anything else, stores the raw event in `webhook_events` and returns
HTTP 200 immediately. Processing happens asynchronously in `billing-worker`.
