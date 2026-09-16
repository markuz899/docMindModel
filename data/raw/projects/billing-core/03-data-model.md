# Data Model

## subscriptions

`subscriptions` holds `id`, `user_id`, `plan_code`, `status`, `current_period_start`,
`current_period_end`, `cancel_at_period_end`. `status` is one of `trialing`,
`active`, `past_due`, `canceled`.

## invoices

`invoices` holds `id`, `subscription_id`, `number`, `currency`, `amount_cents`,
`status`, `issued_at`, `due_at`, `pdf_url`. `status` is one of `draft`, `open`,
`paid`, `uncollectible`, `void`. The `number` column is unique and assigned at
issue time, never at creation time.

## payment_attempts

`payment_attempts` holds `id`, `invoice_id`, `provider_intent_id`, `status`,
`error_code`, `attempt_number`, `created_at`. One row per charge attempt. The
current retry count of an invoice is the count of rows for that invoice.

## webhook_events

`webhook_events` stores every received Stripe event with `provider_event_id`
(unique), `type`, `payload`, `processed_at`. The unique index on
`provider_event_id` is what makes webhook processing idempotent.
