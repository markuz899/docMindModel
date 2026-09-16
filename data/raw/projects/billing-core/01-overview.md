# Billing Core

## Purpose

Billing Core turns usage and subscriptions into invoices and collects payment.
It is a Python 3.11 FastAPI service. It is the only service allowed to talk to
the payment provider; every other service asks Billing Core instead of calling
the provider directly.

## Boundaries

Billing Core owns subscriptions, invoices, payment attempts and refunds. It does
not own user identity: it stores `user_id` as an opaque reference and resolves
display data from the identity service when a human-readable invoice is needed.

## Deployment

The service runs as two workloads from the same image: the HTTP API and the
`billing-worker` process that drains the job queue. Both are deployed together
and share the same configuration.
