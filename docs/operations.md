# SheetSense Operations Runbook

## Release

1. Run formatting, lint, type checking, tests, coverage, browser accessibility,
   write-isolation, dependency, secret, and evaluation-regression gates.
2. Build an immutable image tagged with the Git SHA and retain its SBOM.
3. Run 'alembic upgrade head' as a separate migration job.
4. Deploy to staging, execute smoke and live-agent evaluation, then promote the
   identical image digest to production.

## Rollback

Redeploy the last known-good image digest. Prefer forward corrective migrations
over automatic database downgrades. Never automatically retry an action in
'executing' or 'failed' because its Google Sheets outcome may be unknown.

## Backup, retention, and alerts

- Use daily PostgreSQL snapshots and point-in-time recovery; test restores quarterly.
- Retain audit events 365 days, traces 30 days, and evaluation cases 180 days.
- Alert on write failures, stuck actions, auth spikes, quota exhaustion, Redis
  unavailability, and readiness failures.
- Correlate incidents with request_id and trace_id from response to tool and audit.

## Incident response

Disable confirmations if write integrity is uncertain, preserve audit records,
rotate affected credentials, identify impacted tenants, and document remediation.
