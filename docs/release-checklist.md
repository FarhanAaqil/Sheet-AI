# Release checklist

## Required automated gates

- [ ] Ruff formatting and linting
- [ ] Mypy type checking
- [ ] Unit, integration, concurrency, and property tests
- [ ] 85% line and 75% branch coverage floors
- [ ] Static write-isolation scan
- [ ] Deterministic evaluation regression comparison
- [ ] Repeated live-agent evaluation with approved credentials
- [ ] Adversarial safety evaluation with 100% destructive-action gating
- [ ] npm and Python dependency audits with no unresolved high findings
- [ ] Playwright journeys in Chromium, Firefox, and WebKit
- [ ] axe with no serious or critical violations
- [ ] Responsive verification at 360, 768, 1024, and 1440 pixels
- [ ] Docker image build and liveness smoke test
- [ ] SBOM generation and artifact retention

## Release procedure

1. Run the full CI workflow against the release commit.
2. Build an immutable image tagged with the commit SHA.
3. Run Alembic migrations as a separate staging job.
4. Deploy to staging and verify login, connection, reads, chat, confirmation,
   rejection, conflicts, audit history, and observability.
5. Review live evaluation and latency evidence.
6. Approve production deployment.
7. Run production migrations, deploy, and monitor write failures, auth
   failures, quotas, and stuck actions.

## Rollback

Use the previously signed image, follow the compatibility notes in
`docs/operations.md`, and never downgrade the database before confirming that
the prior application schema supports the current migration state.
