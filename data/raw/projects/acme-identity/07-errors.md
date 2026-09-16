# Errors

## Error envelope

Every error response uses `{ "error": "<code>", "message": "<human text>" }`.
The `error` code is stable and safe to switch on; `message` is not.

## Authentication errors

`invalid_credentials` (HTTP 401) is returned by `POST /auth/login` when the
email is unknown or the password does not match. `invalid_refresh_token`
(HTTP 401) is returned by `POST /auth/refresh`. `token_expired` (HTTP 401) is
returned by `authMiddleware` when the `exp` claim is in the past.

## Authorisation errors

`insufficient_scope` (HTTP 403) is returned when a route requires a scope the
token does not carry, for example calling `GET /users/:id` without
`users:read`.

## Validation errors

`validation_failed` (HTTP 422) is returned when the request body does not match
the expected shape. The `message` field lists the offending fields.
