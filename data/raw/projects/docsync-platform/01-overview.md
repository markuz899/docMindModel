# DocSync Platform

## Purpose

DocSync generates, signs and delivers documents (contracts, certificates,
policy files) to end users. It is a Java 21 Spring Boot service. Documents are
produced from templates, stored in object storage and delivered through one of
several delivery providers.

## Vocabulary

A *document request* is what a caller submits. A *document* is the rendered
artefact. A *delivery* is one attempt to hand a document to a recipient through
a provider. One document can have several deliveries.

## Components at a glance

`DocumentRequestController` accepts requests, `RenderService` produces the
artefact, `SignatureService` seals it, `DeliveryService` hands it to a provider
and `DeliveryStatusListener` consumes provider callbacks.
