# API Reference

## POST /auth/login

Accepts `{ email, password }`. On success returns `{ accessToken, refreshToken,
expiresIn }` with HTTP 200. On invalid credentials returns HTTP 401 with body
`{ "error": "invalid_credentials" }`. The route is rate limited to 10 attempts
per email per 15 minutes.

## POST /auth/refresh

Accepts `{ refreshToken }` and returns a new access token. If the refresh token
is unknown, expired or already revoked, the route returns HTTP 401 with
`{ "error": "invalid_refresh_token" }`.

## GET /me

Returns the authenticated user. The route requires a valid bearer token and is
handled by `UserController.getMe`, which calls `UserService.getCurrentUser`.
The response body is:

```json
{
  "id": "uuid",
  "email": "user@example.com",
  "displayName": "Ada Lovelace",
  "avatarUrl": "https://cdn.acme.dev/avatars/ada.png",
  "locale": "en-GB",
  "status": "active"
}
```

`displayName`, `avatarUrl` and `locale` come from the profile record. `id`,
`email` and `status` come from the account record.

## PATCH /me

Updates profile fields only. Accepts any subset of `displayName`, `avatarUrl`,
`locale`, `timezone`. Account fields such as `email` and `status` are not
editable through this route. Returns the updated `CurrentUser` payload.

## GET /users/:id

Internal route, requires the `users:read` scope. Returns the same shape as
`GET /me` for an arbitrary account id. Used by other services.
