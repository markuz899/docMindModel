# Architecture

## Layers

Requests enter through `AuthController` or `UserController`. Controllers do no
business logic: they validate the request shape and delegate to a service.
Services (`AuthService`, `UserService`, `ProfileService`) hold the business
rules. Repositories (`UserRepository`, `ProfileRepository`, `SessionRepository`)
are the only code allowed to talk to the database.

## UserService

`UserService` is the entry point for everything about a user account. It
composes the account record and the profile record into the `CurrentUser`
object returned by the API. It depends on `UserRepository` for the account row
and on `ProfileService` for profile fields. `UserService` never queries the
database directly.

## ProfileService

`ProfileService` owns profile fields: `displayName`, `avatarUrl`, `locale` and
`timezone`. It reads and writes through `ProfileRepository`. When a profile row
does not exist for an account, `ProfileService` returns a default profile built
from the account email local part.

## AuthService

`AuthService` verifies credentials, issues access and refresh tokens and
revokes sessions. It depends on `UserRepository` to load the account and on
`SessionRepository` to persist refresh tokens. Token signing uses the RS256 key
pair loaded at boot from the secret store.

## Authentication flow

Every protected route passes through `authMiddleware`. The middleware reads the
`Authorization: Bearer <token>` header, verifies the RS256 signature, checks the
`exp` claim and loads the account id from the `sub` claim. It then attaches
`req.auth = { userId, scopes }`. Routes never parse the token themselves.
