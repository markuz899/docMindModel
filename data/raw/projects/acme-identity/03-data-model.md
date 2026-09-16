# Data Model

## users table

The `users` table holds the account record: `id` (uuid, primary key), `email`,
`password_hash`, `status`, `created_at`, `updated_at`. The `email` column has a
unique index. `status` is one of `active`, `pending`, `suspended`.

## user_profiles table

The `user_profiles` table holds profile data: `user_id` (uuid, primary key and
foreign key to `users.id`), `display_name`, `avatar_url`, `locale`, `timezone`,
`updated_at`. There is exactly one profile row per account, created lazily on
first profile write.

## sessions table

The `sessions` table holds refresh tokens: `id`, `user_id`, `refresh_token_hash`,
`user_agent`, `expires_at`, `revoked_at`. Access tokens are never stored; only
refresh tokens are persisted, and only as a hash.

## Naming

Database columns are snake_case. The API layer maps them to camelCase; the
mapping lives in `UserRepository.toDomain` and `ProfileRepository.toDomain`.
`display_name` in the database is `displayName` in the API payload.
