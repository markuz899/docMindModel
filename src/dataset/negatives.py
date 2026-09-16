"""Templates for unanswerable questions.

The single most important behaviour to teach is "the context does not say".
These templates ask about fact *types* no documentation page in the corpus
contains (latency percentiles, cost, ownership, license, ...). Each template
carries probe terms: if any probe term appears in the sampled context the
template is skipped, so a generated "unanswerable" example can never be
accidentally answerable.

TRAIN and EVAL template sets are disjoint on purpose -- otherwise the
hallucination benchmark would only measure memorised phrasings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NegativeTemplate:
    key: str
    question_it: str
    question_en: str
    topic_it: str
    topic_en: str
    need_it: str
    need_en: str
    probes: tuple[str, ...]


TRAIN_TEMPLATES: tuple[NegativeTemplate, ...] = (
    NegativeTemplate(
        "latency", "Qual è la latenza p99 di {subject}?", "What is the p99 latency of {subject}?",
        "la latenza p99", "p99 latency",
        "la documentazione delle metriche o i dati di monitoraggio",
        "the metrics documentation or monitoring data",
        ("p99", "p95", "latenza", "latency", "percentile", "milliseconds of response"),
    ),
    NegativeTemplate(
        "sla", "Qual è lo SLA di disponibilità per {subject}?", "What is the availability SLA for {subject}?",
        "lo SLA di disponibilità", "the availability SLA",
        "il documento di service level o il contratto di servizio",
        "the service level document",
        ("sla", "slo", "uptime", "availability target", "disponibilit"),
    ),
    NegativeTemplate(
        "owner", "Quale team è owner di {subject}?", "Which team owns {subject}?",
        "il team owner", "the owning team",
        "il registro dei servizi o il file dei code owner",
        "the service catalogue or the code owners file",
        ("owner", "team", "maintainer", "squad", "responsabile"),
    ),
    NegativeTemplate(
        "cost", "Quanto costa in infrastruttura {subject}?", "What does {subject} cost to run?",
        "il costo di esercizio", "the running cost",
        "i dati di costo o il budget dell'infrastruttura",
        "infrastructure cost or budget data",
        ("cost", "costo", "budget", "pricing", "prezzo", "eur", "usd"),
    ),
    NegativeTemplate(
        "coverage", "Qual è la copertura dei test di {subject}?", "What is the test coverage of {subject}?",
        "la copertura dei test", "the test coverage",
        "il report di coverage o la configurazione della CI",
        "the coverage report or the CI configuration",
        ("coverage", "copertura", "unit test", "test suite", "pytest", "junit"),
    ),
    NegativeTemplate(
        "rollback", "Qual è la procedura di rollback per {subject}?", "What is the rollback procedure for {subject}?",
        "la procedura di rollback", "the rollback procedure",
        "il runbook operativo",
        "the operational runbook",
        ("rollback", "revert", "runbook", "undo"),
    ),
    NegativeTemplate(
        "oncall", "Chi va contattato in reperibilità per un problema su {subject}?",
        "Who is on call for an incident on {subject}?",
        "la reperibilità", "the on-call rotation",
        "la rotazione on-call o il piano di escalation",
        "the on-call rotation or escalation plan",
        ("on-call", "oncall", "pager", "escalation", "reperibil", "incident"),
    ),
    NegativeTemplate(
        "license", "Con quale licenza è distribuito {subject}?", "Under which license is {subject} distributed?",
        "la licenza", "the license",
        "il file di licenza del repository",
        "the repository license file",
        ("license", "licenza", "apache", "mit", "gpl", "proprietary"),
    ),
    NegativeTemplate(
        "history", "In quale versione è stato introdotto {subject}?", "In which release was {subject} introduced?",
        "la versione di introduzione", "the introducing release",
        "il changelog o la storia del repository",
        "the changelog or the repository history",
        ("changelog", "release", "versione", "version ", "v1.", "v2.", "migration"),
    ),
    NegativeTemplate(
        "capacity", "Quanta memoria consuma {subject} sotto carico?",
        "How much memory does {subject} use under load?",
        "il consumo di memoria", "memory usage",
        "i dati di capacity planning o i limiti di risorsa",
        "capacity planning data or the resource limits",
        ("memory", "memoria", "heap", "mib", "gib", " gb", "resource limit", "request limit"),
    ),
)

EVAL_TEMPLATES: tuple[NegativeTemplate, ...] = (
    NegativeTemplate(
        "root_cause", "Perché {subject} ha smesso di funzionare ieri sera?",
        "Why did {subject} stop working last night?",
        "la causa dell'incidente", "the cause of the incident",
        "i log o il postmortem dell'incidente",
        "the logs or the incident postmortem",
        ("incident", "postmortem", "outage", "ieri", "last night", "downtime"),
    ),
    NegativeTemplate(
        "throughput", "Quante richieste al secondo regge {subject}?",
        "How many requests per second can {subject} sustain?",
        "il throughput sostenibile", "the sustainable throughput",
        "i risultati di un test di carico",
        "load test results",
        ("requests per second", "rps", "throughput", "richieste al secondo", "load test", "benchmark", "rate limit"),
    ),
    NegativeTemplate(
        "backup", "Ogni quanto viene fatto il backup dei dati di {subject}?",
        "How often is the data of {subject} backed up?",
        "la frequenza dei backup", "the backup frequency",
        "il piano di backup e disaster recovery",
        "the backup and disaster recovery plan",
        ("backup", "restore", "disaster", "snapshot", "recovery"),
    ),
    NegativeTemplate(
        "gdpr", "Per quanto tempo vengono conservati i dati personali in {subject}?",
        "How long is personal data retained in {subject}?",
        "il periodo di retention dei dati personali", "the personal data retention period",
        "la documentazione di privacy o la data retention policy",
        "the privacy or data retention documentation",
        ("retention", "gdpr", "privacy", "conservazione", "personal data", "anonymi"),
    ),
    NegativeTemplate(
        "language_runtime", "Quale versione del runtime viene usata in produzione per {subject}?",
        "Which runtime version runs {subject} in production?",
        "la versione del runtime in produzione", "the production runtime version",
        "la definizione di build o il Dockerfile",
        "the build definition or the Dockerfile",
        ("node 2", "python 3", "java 2", "go 1", "runtime version", "dockerfile", "jdk", "version of"),
    ),
    NegativeTemplate(
        "alerting", "Quali alert scattano quando {subject} è in errore?",
        "Which alerts fire when {subject} fails?",
        "gli alert configurati", "the configured alerts",
        "la configurazione di alerting o le dashboard",
        "the alerting configuration or the dashboards",
        ("alert", "pagerduty", "dashboard", "grafana", "notifica", "monitoring"),
    ),
    NegativeTemplate(
        "scaling", "Quante istanze servono per gestire il picco di {subject}?",
        "How many instances are needed for peak traffic on {subject}?",
        "il dimensionamento per il picco", "peak-traffic sizing",
        "i dati di capacity planning",
        "capacity planning data",
        ("replica", "instances", "istanze", "autoscal", "hpa", "peak", "picco", "scaling"),
    ),
    NegativeTemplate(
        "audit", "Chi ha modificato per ultimo {subject}?",
        "Who last modified {subject}?",
        "l'autore dell'ultima modifica", "the last author",
        "la storia del version control",
        "the version control history",
        ("author", "autore", "commit", "git", "modificat", "changed by"),
    ),
    NegativeTemplate(
        "dependency_version", "Quali librerie di terze parti usa {subject}?",
        "Which third-party libraries does {subject} use?",
        "l'elenco delle dipendenze di terze parti", "the third-party dependency list",
        "il manifest delle dipendenze",
        "the dependency manifest",
        ("dependency", "dipendenz", "library", "librer", "package.json", "requirements", "pom.xml", "go.mod", "sdk"),
    ),
    NegativeTemplate(
        "encryption", "Come vengono cifrati i dati a riposo in {subject}?",
        "How is data at rest encrypted in {subject}?",
        "la cifratura dei dati a riposo", "encryption at rest",
        "la documentazione di sicurezza",
        "the security documentation",
        ("encrypt", "cifrat", "at rest", "kms", "aes", "crittograf"),
    ),
)

# MEASURED, do not "improve" without re-running the benchmark: expanding these
# from 3 to 8 structurally different phrasings per language, on this dataset
# size, made the model strictly worse -- refusal on the hallucination benchmark
# fell 0.98 -> 0.47 and false refusals on answerable questions ROSE 0.17 -> 0.30.
# 42 refusal examples over 3 phrasings is ~14 reinforcements each, enough for a
# 0.5B to learn the behaviour; over 36 phrasings it is ~1.2, which is below the
# learning threshold and teaches nothing coherent. The fix for over-refusal is
# variety *with* volume -- teacher-generated refusals at scale
# (scripts/generate_dataset.py --provider anthropic), not the same few examples
# spread thinner. See RESULTS.md, "The over-refusal experiment".
REFUSAL_TEMPLATES_IT = (
    "La documentazione fornita non riporta {topic} di {subject}.\n\n"
    "Le sezioni disponibili descrivono {headings}, ma nessuna contiene questo dato.\n\n"
    "Per rispondere servirebbe {need}.",
    "Non posso rispondere con le fonti fornite: nessuna delle sezioni indica "
    "{topic} di {subject}.\n\n"
    "Quello che è documentato riguarda {headings}, che è un altro aspetto.\n\n"
    "Per rispondere serve una fonte diversa: {need}.",
    "Le sezioni fornite non sono sufficienti: {topic} non compare in nessuna di esse.\n\n"
    "Sono presenti le sezioni {headings}; non c'è alcun riferimento a questo dato "
    "per {subject}.\n\n"
    "Per una risposta affidabile serve una fonte diversa: {need}.",
)

REFUSAL_TEMPLATES_EN = (
    "The provided documentation does not contain {topic} for {subject}.\n\n"
    "The available sections describe {headings}, but none of them states it.\n\n"
    "Answering this would require {need}.",
    "I cannot answer from the supplied sources: none of them specifies {topic}.\n\n"
    "What is documented covers {headings}, which is a different aspect of {subject}.\n\n"
    "You would need {need}.",
    "The supplied sections are not enough: {topic} is not mentioned anywhere in them.\n\n"
    "They document {headings}; there is no reference to this for {subject}.\n\n"
    "A reliable answer would require {need}.",
)
