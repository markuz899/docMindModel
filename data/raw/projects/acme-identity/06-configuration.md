# Configuration

## Sources

Configuration is read at boot by `config/loadConfig.ts`. Values come from
environment variables, with defaults declared in `config/defaults.ts`. There is
no runtime reload: changing configuration requires a restart.

## Token settings

`ACCESS_TOKEN_TTL` controls the access token lifetime and defaults to 15
minutes. `REFRESH_TOKEN_TTL` defaults to 30 days. `JWT_KEY_ID` selects which
RS256 key pair is used for signing.

## Database settings

`DATABASE_URL` is the Postgres connection string. `DB_POOL_SIZE` defaults to 10
connections per replica. `DB_STATEMENT_TIMEOUT_MS` defaults to 5000 and is
applied per statement by the pool.

## HTTP settings

`HTTP_PORT` defaults to 8080. `REQUEST_TIMEOUT_MS` defaults to 30000 and is
enforced by the server, not by the gateway.
