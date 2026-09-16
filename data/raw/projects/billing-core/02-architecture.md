# Architecture

## Components

`InvoiceService` builds invoices from subscription and usage records.
`PaymentService` executes charges and refunds. `StripeGateway` is the only
component that performs HTTP calls to Stripe. `WebhookHandler` consumes Stripe
callbacks. `SubscriptionRepository`, `InvoiceRepository` and `PaymentRepository`
own database access.

## StripeGateway

`StripeGateway` wraps the Stripe SDK. It exposes `create_payment_intent`,
`capture`, `refund` and `retrieve_event`. It is the single integration point
with Stripe: `PaymentService` calls `StripeGateway`, never the SDK. The gateway
adds the idempotency key and translates Stripe errors into domain errors.

## PaymentService

`PaymentService` decides when to charge, how many times to retry and when to
mark an invoice as uncollectible. It depends on `StripeGateway` for the actual
call and on `PaymentRepository` to record each attempt. Retry state lives in the
`payment_attempts` table, not in memory.

## Queue

Long running work is pushed to Redis and consumed by `billing-worker`. Invoice
issuing, dunning emails and webhook replays are queued. The queue is at-least
once: every consumer must be idempotent.
