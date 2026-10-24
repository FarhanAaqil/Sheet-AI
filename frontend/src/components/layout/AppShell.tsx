import {
  Activity,
  Bot,
  ChevronDown,
  Menu,
  PanelRight,
  Settings,
  ShieldCheck,
  Table2
} from "lucide-react";
import type { ReactNode } from "react";

export type WorkspaceView = "data" | "analytics" | "schema" | "audit" | "settings";

type Props = {
  children: ReactNode;
  sidebar: ReactNode;
  assistant: ReactNode;
  connectionName: string;
  activeSheet?: string;
  view: WorkspaceView;
  onViewChange: (view: WorkspaceView) => void;
  sidebarOpen: boolean;
  assistantOpen: boolean;
  onSidebarToggle: () => void;
  onAssistantToggle: () => void;
  connected: boolean;
};

const views: Array<{ id: WorkspaceView; label: string }> = [
  { id: "data", label: "Data" },
  { id: "analytics", label: "Analytics" },
  { id: "schema", label: "Schema" }
];

export function AppShell({
  children,
  sidebar,
  assistant,
  connectionName,
  activeSheet,
  view,
  onViewChange,
  sidebarOpen,
  assistantOpen,
  onSidebarToggle,
  onAssistantToggle,
  connected
}: Props) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-group">
          <button className="icon-button mobile-only" onClick={onSidebarToggle} aria-label="Open sheet browser">
            <Menu size={20} />
          </button>
          <a href="/" className="brand" aria-label="SheetSense home">
            <span className="brand-mark" aria-hidden="true"><Table2 size={20} /></span>
            <span>SheetSense</span>
          </a>
          <span className="product-tag">AI workspace</span>
        </div>

        <div className="workspace-context" aria-label="Current workspace">
          <button className="context-button" type="button">
            <span>
              <small>Workspace</small>
              <strong>{connectionName}</strong>
            </span>
            <ChevronDown size={16} aria-hidden="true" />
          </button>
          <span className={`status-badge ${connected ? "is-live" : "is-offline"}`}>
            <span className="status-dot" aria-hidden="true" />
            {connected ? "Live connection" : "Not connected"}
          </span>
        </div>

        <div className="topbar-actions">
          <button className="icon-button" onClick={() => onViewChange("audit")} aria-label="Open activity history">
            <Activity size={19} />
          </button>
          <button className="icon-button" onClick={() => onViewChange("settings")} aria-label="Open settings">
            <Settings size={19} />
          </button>
          <button
            className="assistant-toggle"
            onClick={onAssistantToggle}
            aria-expanded={assistantOpen}
            aria-label="Assistant"
          >
            <Bot size={18} />
            <span>Assistant</span>
            <PanelRight size={16} />
          </button>
          <div className="avatar" aria-label="Signed in user">FA</div>
        </div>
      </header>

      <div className="viewbar">
        <div className="viewbar-title">
          <span className="sheet-icon"><Table2 size={17} /></span>
          <strong>{activeSheet || "Choose a worksheet"}</strong>
          {activeSheet && <span className="safe-label"><ShieldCheck size={14} /> Protected</span>}
        </div>
        <nav className="view-tabs" aria-label="Workspace views">
          {views.map((item) => (
            <button
              key={item.id}
              className={view === item.id ? "active" : ""}
              onClick={() => onViewChange(item.id)}
              aria-current={view === item.id ? "page" : undefined}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="workspace-frame">
        <div className={`drawer-backdrop ${sidebarOpen ? "visible" : ""}`} onClick={onSidebarToggle} />
        <aside className={`sheet-sidebar ${sidebarOpen ? "open" : ""}`} aria-label="Sheet browser">
          {sidebar}
        </aside>
        <main id="workspace" className="workspace-main">{children}</main>
        <aside className={`assistant-panel ${assistantOpen ? "open" : ""}`} aria-label="AI assistant">
          {assistant}
        </aside>
      </div>
    </div>
  );
}
