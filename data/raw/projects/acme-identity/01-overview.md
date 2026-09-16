# Acme Identity Service

## Purpose

Acme Identity is the internal service that owns user accounts, profiles and
session tokens for every Acme product. It is a TypeScript service running on
Node 20 and exposes a REST API behind the internal gateway. All other services
treat it as the single source of truth for "who is this user".

## Scope

The service owns authentication (password and OIDC), token issuance, profile
data and account lifecycle. It does not own billing, entitlements or
notification preferences; those live in other services and only reference the
user id.

## Runtime

The service runs as a stateless container. Two replicas are deployed per
environment. Sessions are not held in memory: every request is authenticated
from the bearer token, so any replica can serve any request.
