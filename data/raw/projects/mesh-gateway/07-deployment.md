# Deployment

## Kubernetes

The gateway is deployed as a Deployment with 4 replicas and a HorizontalPodAutoscaler
targeting 60% CPU. The Service is of type LoadBalancer. Config is mounted from
the `mesh-gateway-config` ConfigMap; `routes.yaml` lives in the same ConfigMap.

## Rollout

Rollouts are `RollingUpdate` with `maxUnavailable: 0`. A SIGHUP is sent to
reload `routes.yaml` without restarting, but a change to `gateway.yaml` requires
a pod restart.

## Health

`/healthz` is the liveness probe and checks only the process. `/readyz` is the
readiness probe and additionally checks the Redis connection used by
`RateLimiter`.
