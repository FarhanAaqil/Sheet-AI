# SheetSense AI — Four-Phase Production Implementation Plan

## Purpose

This plan upgrades SheetSense AI from a strong prototype into a secure, concurrent, measurable, and production-ready spreadsheet agent. It addresses the complete technical audit and replaces the current dark/demo-oriented interfaces with one polished, accessible light-theme product.

A “10/10” rating cannot be guaranteed by completing a checklist. The target is to remove all known critical and high-risk issues and provide objective evidence for security, reliability, agent quality, accessibility, and deployment readiness.

## Guiding principles

- Security and data isolation take priority over visual polish.
- The product must never present simulated data or actions as live behavior.
- Every destructive action must be authorized, atomic, auditable, and conflict-aware.
- User, session, spreadsheet, and pending-action state must be isolated.
- Documentation and benchmarks must describe what the system actually does.
- One canonical frontend should replace the duplicated static and Streamlit experiences.
- Each phase has a mandatory acceptance gate before the next phase is considered complete.

---

# Phase 1 — Security, safety, and trustworthy behavior

## Objective

Remove release-blocking vulnerabilities, make configuration fail safely, stop sensitive information leakage, and ensure the UI never fabricates live results.

## 1. Eliminate frontend XSS

Affected areas:

- `static/index.html`
- `app.py`
- Future frontend rendering components

Implementation tasks:

- Replace untrusted `innerHTML` rendering with DOM construction and `textContent`.
- Treat sheet names, columns, cells, user messages, model answers, tool names, and action details as untrusted.
- If Markdown is supported, disable raw HTML and sanitize output with DOMPurify.
- Remove inline event handlers and register events with `addEventListener`.
- Escape values used in element IDs, selectors, URLs, and attributes.
- Add a strict Content Security Policy with no inline scripts, `object-src 'none'`, and `frame-ancestors 'none'`.
- Add `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`, and production HSTS.

Required tests:

- Script tags, event-handler images, SVG payloads, quote breaking, malicious sheet names, malicious model output, and malicious user input.
- Each payload must render as harmless text and never execute.

## 2. Make authentication fail closed

Affected files:

- `main.py`
- `.env.example`
- `docker-compose.yml`
- New `auth/` package

Implementation tasks:

- Add explicit `development`, `test`, `staging`, and `production` environments.
- Refuse production startup when authentication, signing secrets, Gemini credentials, or Google credentials are missing or placeholders.
- Replace optional shared-key behavior with hashed API keys tied to principals and roles.
- Support `viewer`, `editor`, `approver`, and `admin` permissions.
- Bind sessions, sheet connections, and pending actions to the authenticated principal.
- Compare secrets in constant time and support key revocation and rotation.
- Restrict CORS through an `ALLOWED_ORIGINS` setting; reject wildcard production configuration.
- Never inject a real API key into browser HTML or persist secrets in local storage.

## 3. Remove deceptive offline behavior

Implementation tasks:

- Remove canned metrics and fake live answers from the standard application path.
- Remove fake pending actions that claim to be stored in SQLite.
- Create a separately enabled demo mode with a permanent “Demo dataset — no live backend” label.
- Disable real confirmation controls in demo mode.
- When the backend is unavailable, show an honest error state with retry and preserve the user draft.
- Keep demo data under fixtures or a dedicated demo service.

## 4. Sanitize errors, logs, and audit data

Affected files:

- `main.py`
- `agent.py`
- `database.py`
- `sheets_writer.py`
- `retry_handler.py`

Implementation tasks:

- Replace raw exception responses with stable public error codes and an `error_id`.
- Add structured JSON logging with request and trace IDs.
- Avoid logging raw queries, cell values, authorization headers, credentials, and complete spreadsheet URLs.
- Expand redaction to cover generic tokens, passwords, Redis URLs, database URLs, Google keys, authorization headers, and query-string secrets.
- Separate operational logs from append-only business audit events.
- Add configurable retention and redaction policies.

## 5. Repair configuration and deployment

Implementation tasks:

- Use `GCP_CREDENTIALS_JSON` as the recommended runtime-secret mechanism.
- Optionally support `GOOGLE_CREDS_PATH` for local development and document it accurately.
- Remove credential copies from the Dockerfile; credentials must never be baked into images.
- Make `pyproject.toml` authoritative and generate a locked deployment dependency file.
- Run the container as a non-root user with explicit writable data directories.
- Add `/health/live` for liveness and `/health/ready` for dependency readiness.
- Add an API container health check, resource limits, and startup configuration validation.

## Phase 1 acceptance gate

- No XSS test payload executes.
- Production cannot start without valid authentication configuration.
- No fake answer, metric, or action is represented as live.
- API responses contain no raw internal exception details.
- A clean-clone Docker image builds and boots successfully.
- The image contains no credentials.
- Security and secret scans report no unresolved critical findings.

---

# Phase 2 — Concurrency, isolation, and data correctness

**Status: Complete (2026-10-22).**

## Objective

Make the backend correct under simultaneous users, multiple workers, duplicate requests, and changing Google Sheet data.

## 1. Make action confirmation atomic and idempotent

Affected files:

- `database.py`
- `main.py`
- `sheets_writer.py`
- Database migrations

Use this lifecycle:

```text
pending -> executing -> confirmed
                    -> failed
pending -> rejected
pending -> expired
```

Implementation tasks:

- Add an atomic `claim_pending_action()` update conditioned on owner, `pending` status, and an unexpired timestamp.
- Continue only when exactly one row was claimed.
- Persist principal, claim time, execution time, failure code, idempotency key, and result summary.
- Authorize confirm/reject operations against the action owner and role.
- Store expected old values or row fingerprints when staging changes.
- Refresh the target before execution and return `409 Conflict` when it changed.
- Prevent row-shift deletion by revalidating the unique identifier and row contents.
- Define recovery rules for actions stuck in `executing`.

Required tests:

- Two simultaneous confirmations result in exactly one write.
- Confirm/reject and confirm/expiry races.
- Cross-user confirmation attempt.
- Duplicate idempotency key.
- Target changed between staging and confirmation.
- Network timeout after a possible Google API write.

## 2. Remove mutable request state from singleton services

Affected file:

- `agent.py`

Implementation tasks:

- Remove `active_session_id`, `last_pending_action`, and the globally mutable active spreadsheet.
- Introduce an immutable request context containing principal, session, spreadsheet connection, and request IDs.
- Pass context explicitly into tool operations.
- Return a pending action in the current tool result instead of storing it on shared `SheetTools` state.
- Keep only concurrency-safe services shared, such as the LLM client and repositories.
- Make spreadsheet connections tenant-scoped and stop rebuilding the global agent when a sheet is connected.

## 3. Fix session isolation

Implementation tasks:

- Require a session ID or generate a cryptographically random server-side ID.
- Remove the shared `__default__` session.
- Namespace Redis keys as `session:{principal_id}:{session_id}`.
- Verify session ownership on every request.
- Use Redis transactions, Lua, or lists/streams for atomic history updates.
- Serialize or safely merge concurrent turns in the same session.
- Add create, list, clear, and delete session APIs with retention rules.

## 4. Add a tenant-scoped spreadsheet registry

Create a persistent connection model containing:

- Connection ID
- Principal or workspace ID
- Spreadsheet ID
- Display name
- Permission level
- Creation and last-sync timestamps

Implementation tasks:

- Validate Google Sheets URLs and store normalized spreadsheet IDs.
- Select a connection ID per request instead of switching global state.
- Check principal access before every read, write, or schema request.
- Minimize Google API scopes and remove Drive scope when unnecessary.
- Avoid returning sensitive source URLs unless required.

## 5. Make data freshness explicit

Implementation tasks:

- Create a cache service with TTL, schema/version metadata, last refresh time, and per-sheet invalidation.
- Support explicit refresh requests.
- Return `last_refreshed_at`, `cache_age_seconds`, and source metadata.
- Refresh destructive targets before checking preconditions.
- Invalidate only affected sheets after writes.
- Rebuild retrieval indexes only when schema changes.

## 6. Correct spreadsheet tools

### Anomaly detection

- Preserve DataFrame index alignment after numeric conversion.
- Handle nulls, text, empty numeric series, and zero-IQR columns.
- Return excluded-value counts and calculated thresholds.

### Read operations

- Replace the ignored natural-language query argument with structured filters, selected columns, sorting, limits, and pagination.
- Return explicit truncation and next-page metadata.
- Never imply that the first 50 rows represent the complete sheet.

### Aggregation

- Validate type compatibility.
- Make `contains` literal by default with `regex=False`.
- Add row, time, and output-size limits.
- Reduce or remove the expression interface unless a validated use case requires it.

### Cross-sheet joins

- Estimate cardinality before merging.
- Add maximum input and output sizes.
- Warn about duplicate join keys.
- Reject explosive joins and use chunked/database-backed execution for large jobs.

### Data API

- Constrain limits, reject negative values, add pagination, and normalize NaN or non-JSON-native values.

## 7. Improve retrieval correctness and privacy

Implementation tasks:

- Exclude zero-similarity candidates and apply a minimum relevance threshold.
- Return no retrieval context when confidence is insufficient.
- Do not send sensitive sample values to the LLM by default.
- Add configurable column classification and redaction.
- Cache deterministic reformulations.
- Record retrieval scores for evaluation.
- Use LLM reformulation only when measured gains justify the latency and cost.

## Phase 2 acceptance gate

- One hundred parallel confirmation attempts produce exactly one write.
- No request receives another request’s session or pending action.
- One user connecting a sheet cannot affect another user.
- External changes appear within the documented freshness window.
- Null-containing anomaly columns work correctly.
- Reads disclose pagination and truncation.
- Oversized joins are safely rejected.
- Unrelated retrieval queries return no candidate context.

Verification evidence:

- The complete automated suite passes.
- Eight dedicated Phase 2 acceptance tests cover 100-way confirmation
  concurrency, action ownership, session isolation, connection isolation,
  cache freshness, null-safe anomaly detection, join-cardinality limits, and
  sensitive schema-value redaction.
- Python compilation and repository diff validation pass.

---

# Phase 3 — Agent quality, testing, observability, and production engineering

**Status: Complete (2026-10-24).**

## Objective

Replace self-validating benchmarks with credible evidence, strengthen the architecture, and make failures diagnosable.

## 1. Replace the canned benchmark

Create three evaluation modes:

1. Deterministic tool evaluation against fixed fixtures.
2. Real-agent integration evaluation with model/version metadata and repeated runs.
3. Adversarial safety evaluation covering prompt injection, formula injection, XSS content, replay, cross-user access, and resource exhaustion.

Measure:

- Tool-selection precision and recall
- Answer correctness and grounding
- Hallucination rate
- Destructive-action gating rate
- Cross-tenant isolation rate
- Retrieval precision at K
- Latency p50, p95, and p99
- Cost per query
- Error and retry rates

Rules:

- Do not describe keyword routing as live model evaluation.
- Keep offline and live latency separate.
- Persist individual failures and aggregate results.
- Compare every run with a versioned baseline.
- Fail CI when safety or quality regressions exceed agreed thresholds.

## 2. Raise test quality and coverage

Targets:

- At least 85% line coverage.
- At least 75% branch coverage.
- More than 90% coverage for confirmation, authorization, and writer modules.

Add tests for:

- Authentication and authorization
- XSS and sanitization
- Session and tenant isolation
- Confirmation concurrency
- Concurrent chats
- Cache freshness
- Null and malformed data
- Pagination and join limits
- Redis and Google API failures
- Database contention
- Docker startup
- Browser workflows
- Accessibility
- Dependency upgrades

Use Hypothesis for property testing, Playwright for browser tests, axe-core for accessibility, and Locust or k6 for load tests.

## 3. Restructure the backend

Target structure:

```text
backend/
  api/
    routes/
    dependencies/
    schemas/
  agent/
    orchestrator.py
    prompts.py
    tools/
    retrieval/
  auth/
  services/
    sheets.py
    sessions.py
    actions.py
    cache.py
  repositories/
    actions.py
    audit.py
    connections.py
  observability/
  settings.py
  main.py
```

Implementation tasks:

- Replace deprecated FastAPI startup events with lifespan handlers.
- Replace deprecated Pydantic field metadata and mutable defaults.
- Add typed domain errors and service/repository boundaries.
- Add Ruff formatting/linting and mypy or pyright type checking.
- Remove outdated comments and dead code.
- Repair encoding corruption throughout source and documentation.

## 4. Adopt a production database strategy

- Keep SQLite for local development only.
- Use PostgreSQL in production.
- Add Alembic migrations.
- Enforce action-state constraints, ownership, foreign keys, and unique idempotency keys.
- Protect append-only audit events.
- Define backups, retention, and archival.

## 5. Add observability

Implementation tasks:

- Add request IDs, trace IDs, OpenTelemetry traces, and Prometheus-compatible metrics.
- Measure LLM tokens/cost, Google API calls, cache behavior, Redis failures, tool results, action conflicts, and retrieval confidence.
- Add alerts for write failures, auth failures, quota exhaustion, and stuck actions.
- Provide a redacted administrator audit view.

## 6. Harden CI/CD

Required CI stages:

1. Formatting and linting
2. Type checking
3. Unit tests
4. Integration and concurrency tests
5. Frontend and accessibility tests
6. Static write-isolation scan
7. Secret and dependency scanning
8. Docker build and startup smoke test
9. Evaluation regression comparison

Release tasks:

- Build immutable versioned images.
- Generate an SBOM.
- Sign images where supported.
- Run migrations separately.
- Deploy to staging before production.
- Document rollback and incident procedures.

## Phase 3 acceptance gate

Completion evidence:

- The versioned offline evaluation baseline passes with no regressions; live and
  adversarial modes persist individual cases, model metadata, latency, quality,
  safety, retry, and error metrics.
- Coverage gates enforce at least 85% line and 75% branch coverage. The completion
  run achieved 87.00% line and 78.92% branch coverage.
- Ruff formatting/linting and mypy checks are enforced in CI.
- PostgreSQL, Alembic migrations, constrained action states, idempotency, and
  append-only audit events are implemented while SQLite remains the local default.
- Request and trace IDs, Prometheus metrics, optional OTLP export, redacted audit
  inspection, and action/tool instrumentation provide end-to-end diagnostics.
- CI includes security and dependency scans, browser accessibility tests, Docker
  build/boot smoke tests, evaluation regression checks, and SBOM generation.

- Coverage targets pass.
- Repeated real-agent evaluations meet published thresholds.
- Safety adherence is 100% for destructive-action cases.
- No canned result is represented as a live benchmark.
- CI builds and boots the production container.
- Type checking and linting are clean.
- A failed request can be traced from request ID through tool execution.

---

# Phase 4 — Complete light-theme UI/UX redesign and release

**Status: Complete (2026-10-24).**

## Objective

Replace the duplicated dark/demo interfaces with one coherent, safe, responsive, and accessible light-theme application.

## 1. Establish one canonical frontend

Recommended implementation:

- Create `frontend/` with React, TypeScript, and Vite.
- Keep FastAPI as the API server.
- Generate a typed client from the OpenAPI schema.
- Retire `static/index.html` after feature parity.
- Move Streamlit to `examples/` or remove it.
- Do not maintain two production interfaces.

## 2. Light visual system

Use the following starting palette:

```text
App canvas:        #F6F8FC
Surface:           #FFFFFF
Raised surface:    #FBFCFE
Primary blue:      #2563EB
Primary hover:     #1D4ED8
Secondary teal:    #0F766E
Accent coral:      #F97316
Success:           #15803D
Warning:           #B45309
Error:             #DC2626
Primary text:      #172033
Secondary text:    #667085
Border:            #DDE3EA
Focus ring:        #60A5FA
```

Design rules:

- White cards on a soft blue-gray canvas.
- Blue for primary actions and navigation.
- Teal for connected/live states.
- Coral only for restrained accents.
- Semantic colors reserved for status.
- No dark navigation rail or dark content panels.
- Meet WCAG AA contrast.
- Use Inter or Geist for UI text and JetBrains Mono for identifiers/data.
- Centralize color, spacing, radius, shadow, typography, motion, and breakpoint tokens with CSS variables.

## 3. Redesign the application shell

Desktop layout:

```text
+------------------------------------------------------------+
| Product header | Workspace | Connection | User menu       |
+---------------+---------------------------+----------------+
| Sheet browser | Data workspace            | AI assistant   |
| and saved     | Grid / Analytics / Schema | and activity   |
| views         |                           |                |
+---------------+---------------------------+----------------+
```

Responsive behavior:

- Sheet browser becomes a drawer on smaller screens.
- Assistant becomes a full-height panel.
- Tables scroll horizontally and support column controls.
- Confirmation becomes a mobile bottom sheet.

## 4. Implement complete product flows

### Onboarding

- Welcome and capability explanation.
- Guided Google Sheet connection.
- Permission explanation and connection test.
- Clearly labeled demo workspace.

### Spreadsheet workspace

- Searchable sheet list.
- Freshness indicator and refresh action.
- Virtualized, paginated data grid.
- Column visibility, sorting, filtering, and schema/type views.
- Designed empty, loading, stale, and error states.

### AI assistant

- Schema-aware suggested questions.
- Visible active sheet and freshness.
- Rows scanned, truncation, tools used, and data sources.
- Expandable “How this answer was produced” panel.
- Clear separation between explanation, data, warning, and mutation proposal.
- Safe Markdown rendering with no raw model HTML.

### Confirmation experience

- Show spreadsheet, worksheet, identifier, current value, proposed value, freshness, expiry, and approver.
- Warn on conflicts and require restaging.
- Make deletes visually distinct from ordinary updates.
- Require typed confirmation for high-impact batch actions.
- Show final execution status with an audit-event link.

### Analytics

- Let users choose dimensions and measures.
- Use accessible chart colors and provide a table alternative.
- Label aggregations and filters explicitly.
- Avoid misleading axes and guessed visualizations.

### Audit and settings

- Add activity history for pending, confirmed, rejected, expired, and failed actions.
- Show actor, timestamp, target, result, and request ID.
- Add connection, API-key, member/role, retention, and model settings.

## 5. Accessibility requirements

- Complete keyboard navigation and visible focus states.
- Correct landmarks, labels, and dialog focus management.
- Screen-reader announcements for query completion, errors, confirmation, and expiration.
- Do not communicate status by color alone.
- Respect reduced-motion preferences.
- Use appropriate touch-target sizes.
- Keep the virtualized grid keyboard accessible.

## 6. Frontend architecture

```text
frontend/src/
  app/
  components/
    layout/
    data-grid/
    assistant/
    actions/
    charts/
    feedback/
  features/
    auth/
    connections/
    sheets/
    chat/
    audit/
    settings/
  api/
  hooks/
  styles/
  test/
```

Technical choices:

- TanStack Query for server state.
- Local React state or a small store for UI state.
- Schema-based form validation.
- Centralized API error mapping.
- Secure cookies or short-lived tokens rather than local-storage secrets.
- Abort stale requests when users switch sheets.

## 7. Browser and visual testing

Add Playwright journeys for authentication, connection, browsing, filtering, chat, staging, confirmation, rejection, expiry, conflicts, backend failure, isolation, and malicious content.

Run visual tests at 360px, 768px, 1024px, and 1440px in Chromium, Firefox, and WebKit. Include high zoom, large text, empty states, failures, loading, and unusually long content.

## Phase 4 acceptance gate

Implementation evidence:

- React, TypeScript, Vite, TanStack Query, and a typed API client now provide
  the only production interface; the static CDN page and Streamlit duplicate
  have been removed.
- The approved light palette, bundled fonts, responsive three-panel shell,
  mobile drawers, bottom-sheet confirmation, focus states, reduced motion, and
  screen-reader announcements are implemented.
- Onboarding, worksheet data, schema, analytics, grounded chat evidence,
  mutation review, audit history, connections, and settings use real API
  behavior with explicit loading, empty, stale, and error states.
- The production frontend compiles successfully with patched dependencies and
  zero npm audit findings.
- CI builds the client, exercises Chromium, Firefox, and WebKit journeys, runs
  axe checks, boots the Docker image, and retains release evidence.

- The canonical interface uses the approved light theme with no dark content surfaces.
- Lighthouse accessibility is at least 95.
- axe reports no serious accessibility violations.
- All untrusted content is rendered safely.
- Responsive layouts work at all target widths.
- No fake offline metrics or actions remain.
- Every destructive action has an accurate diff and audit trail.
- Cross-browser end-to-end tests pass.
- Legacy static and Streamlit interfaces are removed or explicitly marked non-production.

---

# Final release definition

The project is ready for a new 10/10 review only when:

- All four phase acceptance gates pass.
- No known critical or high security issues remain.
- Concurrent destructive actions are demonstrably safe and idempotent.
- Users, sessions, sheets, and actions are tenant-isolated.
- Production deployment succeeds from a clean clone.
- Documentation exactly matches runtime behavior.
- Real-agent evaluations replace canned quality claims.
- The UI is light, accessible, responsive, secure, and honest about system state.
- CI prevents regressions across security, concurrency, data correctness, agent quality, deployment, and UX.
