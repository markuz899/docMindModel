# Timeout Policy

## Default upstream timeout

The platform timeout policy sets the default upstream timeout to 10000 ms. Mesh
Gateway applies this default to every route that does not declare its own
`timeoutMs`.

## Client timeouts

Clients are expected to use a timeout larger than the gateway timeout, otherwise
they will give up while the gateway is still waiting for the upstream.
