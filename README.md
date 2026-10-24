# SheetSense AI

SheetSense is a production-oriented conversational workspace for live Google
Sheets. It combines a typed FastAPI service, tenant-scoped spreadsheet access,
grounded agent tools, and a human approval gate for every destructive action.

The canonical interface is a responsive React application with a light,
WCAG-conscious visual system. It never presents fixture data or simulated
actions as live behavior.

## What is included

- Live worksheet browsing, pagination, filtering, sorting, schema inspection,
  freshness metadata, and column visibility controls.
- Grounded AI answers with visible tools, sources, scanned-row information, and
  truncation status.
- Atomic, idempotent confirmation and rejection of updates and deletions.
- Exact action diffs, typed confirmation for deletes, conflict detection, and
  append-only audit events.
- Tenant-owned connections, role-based API keys, isolated sessions, Redis
  support, and PostgreSQL production persistence.
- Offline, live-agent, and adversarial evaluation modes with versioned
  baselines.
- Request and trace IDs, Prometheus metrics, optional OTLP export, redacted
  logging, and administrator audit history.
- Ruff, mypy, pytest coverage gates, Playwright/axe journeys, dependency and
  secret scanning, Docker boot verification, and SBOM generation in CI.

## Quick start

Requirements:

- Python 3.11 or newer
- Node.js 22 or newer
- Google service-account credentials with access to the target sheet
- A Gemini API key
- Redis and PostgreSQL for the production topology

Create local configuration:

```bash
cp .env.example .env
```

Install and build:

```bash
pip install -e ".[dev]"
npm ci --prefix frontend
npm run build --prefix frontend
```

Start the API and bundled client:

```bash
uvicorn main:app --reload --port 8000
```

Open:

- Application: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs
- Liveness: http://127.0.0.1:8000/health/live
- Readiness: http://127.0.0.1:8000/health/ready

For frontend development with hot reload:

```bash
npm run dev --prefix frontend
```

Vite proxies requests beginning with `/api` to the FastAPI development server.
The production client uses same-origin API requests.

## Docker deployment

```bash
docker compose up --build
```

Compose starts:

- The SheetSense API and built React client
- Redis for distributed sessions and rate limiting
- PostgreSQL for production persistence
- A one-shot Alembic migration service

The application is exposed at http://localhost:3000/. Credentials are mounted
or injected at runtime and are never copied into the image.

## Configuration

The annotated template is [.env.example](.env.example). Important production
settings include:

| Variable | Purpose |
| --- | --- |
| `APP_ENV` | `development`, `test`, `staging`, or `production` |
| `GEMINI_API_KEY` | Gemini access credential |
| `GCP_CREDENTIALS_JSON` | Recommended runtime Google credential payload |
| `GOOGLE_CREDS_PATH` | Local-only service-account file path |
| `DATABASE_URL` | PostgreSQL URL; mandatory in production |
| `REDIS_URL` | Distributed session and rate-limit store |
| `SHEETSENSE_API_KEY_HASH` | SHA-256 API-key digest |
| `SHEETSENSE_API_KEYS_JSON` | Multi-principal role-bearing key definitions |
| `ALLOWED_ORIGINS` | Explicit browser origins |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Optional OpenTelemetry collector |

The browser never stores API keys in local storage. The onboarding key exists
only in React memory for the active tab.

## Architecture

```text
React + TypeScript
        |
        v
FastAPI authentication, rate limits, request tracing
        |
        +--> tenant-scoped connections and sessions
        +--> SheetSense agent and read-only analysis tools
        +--> pending action repository
                   |
                   v
             explicit approval
                   |
                   v
        isolated sheets_writer execution
                   |
                   v
         append-only audit event
```

Reasoning code cannot call `gspread` write methods. Only
`sheets_writer.py`, reached after an atomic action claim, can execute a
mutation.

## Frontend behavior

The application provides:

- Guided connection onboarding and honest unconfigured/error states.
- A desktop three-panel workspace that adapts into drawers and full-height
  panels on tablet and mobile.
- An accessible data table with keyboard focus, pagination, and horizontal
  overflow.
- Configurable analytics with explicit dimensions, measures, aggregations, and
  a table alternative.
- Plain-text model rendering—raw model HTML is never injected into the DOM.
- A mobile bottom-sheet confirmation flow with exact current and proposed
  values.
- Reduced-motion support, visible focus, screen-reader status announcements,
  semantic landmarks, and touch-friendly controls.

See [docs/frontend.md](docs/frontend.md) for frontend maintenance details.

## Quality and evaluation

```bash
ruff format --check .
ruff check .
mypy
pytest tests --cov=. --cov-branch
python scripts/check_coverage.py
python scripts/check_write_isolation.py
python scripts/check_eval_regression.py
npm run test:browser
```

Current enforced coverage floors are 85% line and 75% branch. Live-agent
evaluation is intentionally separate from deterministic offline evaluation and
records model/version metadata and repeated cases.

## Database migrations

Local development defaults to SQLite. Production requires PostgreSQL:

```bash
alembic upgrade head
```

Migrations run as a separate Compose service. Do not run application replicas
as migration workers.

## Operations

- Deployment, backup, retention, rollback, alerting, and incident procedures:
  [docs/operations.md](docs/operations.md)
- Final release checklist: [docs/release-checklist.md](docs/release-checklist.md)
- Four-phase implementation record:
  [implementation_plan.md](implementation_plan.md)

## Known boundaries

- SheetSense does not bypass Google permissions.
- Large datasets are paginated; the UI labels page-scoped analytics explicitly.
- A network timeout during a write is reported as an unknown state and requires
  audit review instead of an unsafe automatic retry.
- Live-agent evaluation consumes model quota and must run with approved
  credentials.

## License

[MIT](LICENSE)
