# Main Flows

## Current User Flow

1. The client calls `GET /me` with a bearer token.
2. `authMiddleware` verifies the token and sets `req.auth.userId`.
3. `UserController.getMe` calls `UserService.getCurrentUser(userId)`.
4. `UserService` loads the account through `UserRepository.findById`.
5. `UserService` loads profile fields through `ProfileService.getProfile`.
6. The two records are merged into `CurrentUser` and serialised.

If the profile row is missing, step 5 returns the default profile, so
`displayName` falls back to the email local part.

## Login Flow

1. `POST /auth/login` reaches `AuthController.login`.
2. `AuthService.authenticate` loads the account by email via `UserRepository`.
3. The password is verified against `password_hash` with argon2id.
4. On success `AuthService` issues an access token and a refresh token, and
   stores the refresh token hash through `SessionRepository.create`.

## Profile Update Flow

1. `PATCH /me` reaches `UserController.patchMe`.
2. `ProfileService.updateProfile` writes the changed fields through
   `ProfileRepository.upsert`, creating the row if it does not exist.
3. `UserService.getCurrentUser` is called again to build the response.
