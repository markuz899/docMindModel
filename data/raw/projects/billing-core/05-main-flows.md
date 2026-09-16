# Main Flows

## Invoice issuing flow

1. `POST /invoices/{id}/issue` reaches `InvoiceService.issue`.
2. The invoice status is checked; only `draft` can be issued.
3. `InvoiceRepository.assign_number` allocates the next invoice number.
4. The PDF is rendered and uploaded; `pdf_url` is stored.
5. A `charge_invoice` job is pushed to the queue.
6. `billing-worker` picks up the job and calls `PaymentService.charge`.

## Charge flow

1. `PaymentService.charge` loads the invoice and the previous attempts.
2. `StripeGateway.create_payment_intent` is called with an idempotency key
   derived from `invoice_id` and `attempt_number`.
3. The attempt row is written to `payment_attempts` before the call returns.
4. Stripe confirms asynchronously; the outcome arrives via webhook.

## Webhook flow

1. Stripe posts to `POST /webhooks/stripe`.
2. `WebhookHandler.receive` verifies the signature and inserts into
   `webhook_events`; a duplicate `provider_event_id` is ignored.
3. A `process_event` job is queued.
4. `billing-worker` maps `payment_intent.succeeded` to invoice `paid` and
   `payment_intent.payment_failed` to a new retry decision.
