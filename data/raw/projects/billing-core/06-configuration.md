# Configuration

## Loading

Configuration is a pydantic `Settings` object in `billing/config.py`, populated
from environment variables. Unknown variables are ignored; missing required
variables abort startup.

## Provider settings

`STRIPE_API_KEY` and `STRIPE_WEBHOOK_SECRET` are required. `STRIPE_TIMEOUT_MS`
defaults to 8000 and is passed to the Stripe SDK by `StripeGateway`.
`STRIPE_MAX_NETWORK_RETRIES` defaults to 2 and is handled by the SDK itself.

## Retry settings

`PAYMENT_MAX_ATTEMPTS` defaults to 4. `PAYMENT_RETRY_BACKOFF_HOURS` defaults to
`[1, 24, 72]`, so the second attempt happens one hour after the first, the third
a day later and the fourth three days later.

## Queue settings

`REDIS_URL` is required. `WORKER_CONCURRENCY` defaults to 4.
`JOB_VISIBILITY_TIMEOUT_S` defaults to 300.
