import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  AlertCircle,
  Braces,
  ChevronRight,
  Database,
  KeyRound,
  Link2,
  Plus,
  RefreshCw,
  Search,
  Settings as SettingsIcon,
  Sheet,
  ShieldCheck
} from "lucide-react";
import { useMemo, useState } from "react";
import { ApiError, createApiClient } from "../api/client";
import type { PendingAction } from "../api/types";
import { ConfirmationDialog } from "../components/actions/ConfirmationDialog";
import { AssistantPanel, type Message } from "../components/assistant/AssistantPanel";
import { AnalyticsPanel } from "../components/charts/AnalyticsPanel";
import { DataGrid } from "../components/data-grid/DataGrid";
import { Onboarding } from "../components/feedback/Onboarding";
import { AppShell, type WorkspaceView } from "../components/layout/AppShell";

const PAGE_SIZE = 100;
const messageFor = (error: unknown) =>
  error instanceof ApiError ? `${error.message}${error.errorId ? ` (reference ${error.errorId})` : ""}` : "Something went wrong. Try again.";

export function App() {
  const queryClient = useQueryClient();
  const [entered, setEntered] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [view, setView] = useState<WorkspaceView>("data");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [assistantOpen, setAssistantOpen] = useState(true);
  const [connectionId, setConnectionId] = useState("");
  const [activeSheet, setActiveSheet] = useState("");
  const [page, setPage] = useState(0);
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [pendingAction, setPendingAction] = useState<PendingAction>();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [actionResult, setActionResult] = useState("");
  const [sheetFilter, setSheetFilter] = useState("");
  const [connectUrl, setConnectUrl] = useState("");
  const [onboardingBusy, setOnboardingBusy] = useState(false);
  const [onboardingError, setOnboardingError] = useState("");

  const api = useMemo(() => createApiClient({ apiKey }), [apiKey]);

  const connections = useQuery({
    queryKey: ["connections", apiKey],
    queryFn: ({ signal }) => api.connections(signal),
    enabled: entered
  });
  const selectedConnection = connections.data?.connections.find((item) => item.connection_id === connectionId)
    ?? connections.data?.connections[0];
  const effectiveConnectionId = connectionId || selectedConnection?.connection_id || "";

  const sheets = useQuery({
    queryKey: ["sheets", apiKey, effectiveConnectionId],
    queryFn: ({ signal }) => api.sheets(effectiveConnectionId || undefined, signal),
    enabled: entered,
    select: (data) => data.sheets
  });
  const currentSheet = activeSheet || sheets.data?.[0] || "";

  const sheetData = useQuery({
    queryKey: ["sheet-data", apiKey, effectiveConnectionId, currentSheet, page],
    queryFn: ({ signal }) =>
      api.data(
        currentSheet,
        { connectionId: effectiveConnectionId || undefined, offset: page * PAGE_SIZE, limit: PAGE_SIZE },
        signal
      ),
    enabled: entered && Boolean(currentSheet)
  });

  const schema = useQuery({
    queryKey: ["schema", apiKey, effectiveConnectionId, currentSheet],
    queryFn: ({ signal }) => api.schema(currentSheet, effectiveConnectionId || undefined, signal),
    enabled: entered && view === "schema" && Boolean(currentSheet)
  });

  const audit = useQuery({
    queryKey: ["audit", apiKey],
    queryFn: ({ signal }) => api.audit(signal),
    enabled: entered && view === "audit"
  });

  const connect = useMutation({
    mutationFn: (url: string) => api.connect(url),
    onSuccess: async (result) => {
      setConnectionId(result.connection_id);
      setConnectUrl("");
      await queryClient.invalidateQueries({ queryKey: ["connections"] });
      await queryClient.invalidateQueries({ queryKey: ["sheets"] });
    }
  });

  const chat = useMutation({
    mutationFn: async (message: string) => {
      let activeSession = sessionId;
      if (!activeSession) {
        const created = await api.createSession();
        activeSession = created.session_id;
        setSessionId(activeSession);
      }
      return api.chat({
        message,
        session_id: activeSession,
        connection_id: effectiveConnectionId || undefined,
        sheet_name: currentSheet || undefined
      });
    },
    onSuccess: (result) => {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: result.answer,
          tools: result.tool_calls_made,
          source: currentSheet,
          rowsScanned: sheetData.data?.returned_rows,
          truncated: sheetData.data?.truncated
        }
      ]);
      setPendingAction(result.pending_action);
      if (result.pending_action) setConfirmOpen(true);
    }
  });

  const confirm = useMutation({
    mutationFn: () => api.confirm(pendingAction!.action_id),
    onSuccess: async () => {
      setActionResult("confirmed");
      await queryClient.invalidateQueries({ queryKey: ["sheet-data"] });
      await queryClient.invalidateQueries({ queryKey: ["audit"] });
    }
  });
  const reject = useMutation({
    mutationFn: () => api.reject(pendingAction!.action_id),
    onSuccess: async () => {
      setActionResult("rejected");
      await queryClient.invalidateQueries({ queryKey: ["audit"] });
    }
  });

  const enterWorkspace = (key: string) => {
    setApiKey(key);
    setEntered(true);
  };

  const connectFromOnboarding = async (key: string, url: string) => {
    setOnboardingBusy(true);
    setOnboardingError("");
    try {
      const onboardingApi = createApiClient({ apiKey: key });
      const result = await onboardingApi.connect(url);
      setApiKey(key);
      setConnectionId(result.connection_id);
      setEntered(true);
    } catch (error) {
      setOnboardingError(messageFor(error));
    } finally {
      setOnboardingBusy(false);
    }
  };

  const sendMessage = (text: string) => {
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", text }]);
    chat.mutate(text);
  };

  if (!entered) {
    return (
      <Onboarding
        busy={onboardingBusy}
        error={onboardingError || undefined}
        onConnect={connectFromOnboarding}
        onContinue={enterWorkspace}
      />
    );
  }

  const filteredSheets = (sheets.data ?? []).filter((name) =>
    name.toLocaleLowerCase().includes(sheetFilter.toLocaleLowerCase())
  );
  const connectionName = selectedConnection?.display_name || "My workspace";

  const sidebar = (
    <div className="sidebar-inner">
      <div className="sidebar-heading"><span>Worksheets</span><button className="icon-button" aria-label="Add connection" onClick={() => setView("settings")}><Plus size={17} /></button></div>
      <div className="sidebar-search"><Search size={15} /><input value={sheetFilter} onChange={(e) => setSheetFilter(e.target.value)} placeholder="Find a sheet" aria-label="Find a sheet" /></div>
      <nav className="sheet-list" aria-label="Worksheets">
        {sheets.isLoading && Array.from({ length: 4 }).map((_, index) => <span className="sheet-skeleton" key={index} />)}
        {filteredSheets.map((name) => (
          <button
            key={name}
            className={currentSheet === name ? "active" : ""}
            onClick={() => { setActiveSheet(name); setPage(0); setSidebarOpen(false); }}
          >
            <Sheet size={16} /><span>{name}</span><ChevronRight size={14} />
          </button>
        ))}
        {!sheets.isLoading && !filteredSheets.length && <p className="sidebar-empty">No worksheets available.</p>}
      </nav>
      <div className="sidebar-section">
        <span>Saved views</span>
        <button onClick={() => setView("audit")}><Activity size={16} /> Activity history</button>
        <button onClick={() => setView("settings")}><SettingsIcon size={16} /> Workspace settings</button>
      </div>
      <div className="sidebar-footer">
        <div><span className="connection-dot" /><div><strong>{connectionName}</strong><span>{selectedConnection?.permission_level || "Role-based access"}</span></div></div>
        <button className="icon-button" onClick={() => connections.refetch()} aria-label="Refresh connections"><RefreshCw size={15} /></button>
      </div>
    </div>
  );

  const assistant = (
    <AssistantPanel
      sheet={currentSheet}
      freshness={sheetData.data?.last_refreshed_at ? new Date(sheetData.data.last_refreshed_at).toLocaleTimeString() : undefined}
      messages={messages}
      busy={chat.isPending}
      error={chat.error ? messageFor(chat.error) : undefined}
      pending={pendingAction}
      onSend={sendMessage}
      onReviewAction={() => setConfirmOpen(true)}
    />
  );

  return (
    <>
      <AppShell
        sidebar={sidebar}
        assistant={assistant}
        connectionName={connectionName}
        activeSheet={currentSheet}
        view={view}
        onViewChange={setView}
        sidebarOpen={sidebarOpen}
        assistantOpen={assistantOpen}
        onSidebarToggle={() => setSidebarOpen((value) => !value)}
        onAssistantToggle={() => setAssistantOpen((value) => !value)}
        connected={Boolean(currentSheet)}
      >
        {view === "data" && (
          <DataGrid
            result={sheetData.data}
            loading={sheetData.isLoading}
            error={sheetData.error ? messageFor(sheetData.error) : undefined}
            page={page}
            pageSize={PAGE_SIZE}
            onPageChange={setPage}
            onRefresh={() => sheetData.refetch()}
          />
        )}
        {view === "analytics" && <AnalyticsPanel result={sheetData.data} />}
        {view === "schema" && <SchemaView data={schema.data} loading={schema.isLoading} error={schema.error ? messageFor(schema.error) : undefined} />}
        {view === "audit" && <AuditView events={audit.data?.events ?? []} loading={audit.isLoading} error={audit.error ? messageFor(audit.error) : undefined} />}
        {view === "settings" && (
          <SettingsView
            apiKey={apiKey}
            setApiKey={setApiKey}
            connectionId={effectiveConnectionId}
            connections={connections.data?.connections ?? []}
            setConnectionId={(id) => { setConnectionId(id); setActiveSheet(""); }}
            connectUrl={connectUrl}
            setConnectUrl={setConnectUrl}
            connect={() => connect.mutate(connectUrl)}
            busy={connect.isPending}
            error={connect.error ? messageFor(connect.error) : undefined}
          />
        )}
      </AppShell>
      {confirmOpen && (
        <ConfirmationDialog
          action={pendingAction}
          sheet={currentSheet}
          busy={confirm.isPending || reject.isPending}
          result={actionResult}
          error={confirm.error ? messageFor(confirm.error) : reject.error ? messageFor(reject.error) : undefined}
          onConfirm={() => confirm.mutate()}
          onReject={() => reject.mutate()}
          onClose={() => { setConfirmOpen(false); if (actionResult) { setPendingAction(undefined); setActionResult(""); } }}
        />
      )}
      <div className="sr-only" aria-live="assertive">
        {chat.isSuccess ? "Assistant response complete." : ""}
        {confirm.isSuccess ? "Action confirmed and recorded." : ""}
        {reject.isSuccess ? "Action rejected." : ""}
      </div>
    </>
  );
}

function SchemaView({ data, loading, error }: { data?: Record<string, unknown>; loading: boolean; error?: string }) {
  return (
    <section className="standard-page">
      <div className="page-heading"><div><p className="section-kicker">Data dictionary</p><h1>Worksheet schema</h1></div><Braces size={24} /></div>
      {loading ? <div className="loading-card">Loading schema…</div> : error ? <div className="inline-alert error">{error}</div> : (
        <div className="schema-card"><pre>{JSON.stringify(data ?? {}, null, 2)}</pre></div>
      )}
    </section>
  );
}

function AuditView({ events, loading, error }: { events: Array<Record<string, unknown>>; loading: boolean; error?: string }) {
  return (
    <section className="standard-page">
      <div className="page-heading"><div><p className="section-kicker">Traceable operations</p><h1>Activity history</h1></div><Activity size={24} /></div>
      {error && <div className="inline-alert error"><AlertCircle size={17} />{error}</div>}
      <div className="audit-list">
        {loading && <div className="loading-card">Loading activity…</div>}
        {!loading && !events.length && <div className="state-panel"><Activity size={28} /><h2>No audit events yet</h2><p>Confirmed, rejected, expired, conflicted, and failed actions appear here.</p></div>}
        {events.map((event, index) => (
          <article key={String(event.id ?? index)}>
            <span className={`audit-icon ${String(event.outcome ?? "")}`}><ShieldCheck size={17} /></span>
            <div><strong>{String(event.event_type ?? "Activity event")}</strong><span>{String(event.resource_id ?? "Workspace")}</span></div>
            <div><span className="outcome-label">{String(event.outcome ?? "recorded")}</span><time>{event.created_at ? new Date(String(event.created_at)).toLocaleString() : "Time unavailable"}</time></div>
            <code>{String(event.request_id ?? "No request ID")}</code>
          </article>
        ))}
      </div>
    </section>
  );
}

type SettingsProps = {
  apiKey: string;
  setApiKey: (value: string) => void;
  connectionId: string;
  connections: Array<{ connection_id: string; display_name?: string }>;
  setConnectionId: (value: string) => void;
  connectUrl: string;
  setConnectUrl: (value: string) => void;
  connect: () => void;
  busy: boolean;
  error?: string;
};

function SettingsView(props: SettingsProps) {
  return (
    <section className="standard-page">
      <div className="page-heading"><div><p className="section-kicker">Workspace administration</p><h1>Settings</h1></div><SettingsIcon size={24} /></div>
      <div className="settings-grid">
        <section className="settings-card"><header><KeyRound size={19} /><div><h2>Access key</h2><p>Held in memory for this browser tab only.</p></div></header><label>API key<input type="password" value={props.apiKey} onChange={(e) => props.setApiKey(e.target.value)} autoComplete="off" /></label></section>
        <section className="settings-card"><header><Database size={19} /><div><h2>Active connection</h2><p>Select a tenant-owned spreadsheet connection.</p></div></header><label>Connection<select value={props.connectionId} onChange={(e) => props.setConnectionId(e.target.value)}>{props.connections.map(item => <option key={item.connection_id} value={item.connection_id}>{item.display_name || item.connection_id}</option>)}</select></label></section>
        <section className="settings-card span-two"><header><Link2 size={19} /><div><h2>Connect another Google Sheet</h2><p>Requires an administrator role and access through the configured service account.</p></div></header><div className="inline-form"><input value={props.connectUrl} onChange={(e) => props.setConnectUrl(e.target.value)} placeholder="https://docs.google.com/spreadsheets/d/…" /><button className="primary-button" onClick={props.connect} disabled={props.busy || !props.connectUrl}>{props.busy ? "Connecting…" : "Test and connect"}</button></div>{props.error && <div className="inline-alert error">{props.error}</div>}</section>
        <section className="settings-card"><header><ShieldCheck size={19} /><div><h2>Roles and approvals</h2><p>Viewer, editor, approver, and administrator permissions are enforced by the API.</p></div></header><span className="settings-status">Managed by API key policy</span></section>
        <section className="settings-card"><header><Activity size={19} /><div><h2>Retention and model</h2><p>Runtime retention, audit, and model settings are controlled by deployment configuration.</p></div></header><span className="settings-status">Environment managed</span></section>
      </div>
    </section>
  );
}
