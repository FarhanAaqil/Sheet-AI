# Frontend maintenance

The only production frontend is `frontend/`, built with React, TypeScript,
Vite, TanStack Query, and Zod. FastAPI serves `frontend/dist/index.html` at
`/` and its immutable assets at `/assets`.

## Structure

```text
frontend/src/
  app/                  Application composition and server-state wiring
  api/                  Typed API client, error mapping, and contracts
  components/
    actions/            Human approval and action diffs
    assistant/          Grounded conversation interface
    charts/             Explicit analytics and table alternative
    data-grid/          Paginated worksheet table
    feedback/           Onboarding and state feedback
    layout/             Responsive product shell
  styles/               Tokens, components, and responsive rules
```

## Security rules

- Never use `dangerouslySetInnerHTML` for user, sheet, cell, or model content.
- Do not store API keys in local storage, session storage, URLs, or build-time
  variables.
- Every mutation must originate from a backend-issued pending action.
- Confirmation UI must show current/expected state and proposed state.
- Do not fabricate live metrics, worksheet rows, connection states, or actions.
- AbortSignal from TanStack Query must be passed into fetch calls.

## Design system

Tokens live in `src/styles/tokens.css`. The interface deliberately uses light
surfaces only:

- Canvas `#F6F8FC`
- Surface `#FFFFFF`
- Primary `#2563EB`
- Teal `#0F766E`
- Coral `#F97316`
- Text `#172033`
- Border `#DDE3EA`
- Focus `#60A5FA`

Inter and JetBrains Mono are bundled through Fontsource; the production client
does not depend on third-party font CDNs.

## Build

```bash
npm ci --prefix frontend
npm run build --prefix frontend
```

The Dockerfile builds the client in its own Node stage. Do not commit
`node_modules` or `frontend/dist`.
