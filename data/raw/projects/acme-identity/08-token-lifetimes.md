# Token Lifetimes

## Access tokens

Access tokens are short lived. The platform standard is a 60 minute access
token lifetime, and Acme Identity follows the platform standard.

## Refresh tokens

Refresh tokens live for 30 days and are rotated on every use: the previous
refresh token is revoked as soon as a new one is issued.
