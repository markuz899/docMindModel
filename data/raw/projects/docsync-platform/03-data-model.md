# Data Model

## document_requests

`document_requests` holds `id`, `tenant_id`, `template_code`, `payload_json`,
`channel`, `status`, `created_at`. `status` is one of `received`, `rendering`,
`render_failed`, `rendered`, `delivering`, `delivered`, `failed`.

## documents

`documents` holds `id`, `request_id`, `storage_key`, `checksum_sha256`,
`signed`, `size_bytes`, `created_at`. One row per successfully rendered
artefact.

## deliveries

`deliveries` holds `id`, `document_id`, `provider`, `recipient`, `status`,
`provider_message_id`, `error_code`, `attempt_number`, `created_at`,
`updated_at`. `status` is one of `queued`, `sent`, `accepted`, `rejected`,
`bounced`.

## templates

`templates` holds `code` (primary key), `version`, `engine`, `body`,
`active`. Only rows with `active = true` can be resolved by `RenderService`.
