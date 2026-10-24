import { AlertTriangle, ArrowRight, CheckCircle2, Clock3, ShieldCheck, Trash2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { PendingAction } from "../../api/types";

type Props = {
  action?: PendingAction;
  sheet?: string;
  busy: boolean;
  result?: string;
  error?: string;
  onConfirm: () => void;
  onReject: () => void;
  onClose: () => void;
};

function EntryList({ value }: { value?: Record<string, unknown> }) {
  if (!value || !Object.keys(value).length) return <span className="muted">Not provided by the action</span>;
  return (
    <dl className="diff-list">
      {Object.entries(value).map(([key, item]) => (
        <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{String(item ?? "Empty")}</dd></div>
      ))}
    </dl>
  );
}

export function ConfirmationDialog({ action, sheet, busy, result, error, onConfirm, onReject, onClose }: Props) {
  const dialog = useRef<HTMLDivElement>(null);
  const destructive = Boolean(action?.tool_name?.toLocaleLowerCase().includes("delete"));
  const [typed, setTyped] = useState("");
  useEffect(() => {
    if (action) dialog.current?.focus();
  }, [action]);
  if (!action) return null;

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className={`confirmation-dialog ${destructive ? "danger" : ""}`} role="dialog" aria-modal="true" aria-labelledby="confirmation-title" tabIndex={-1} ref={dialog}>
        <header>
          <span className="dialog-icon">{destructive ? <Trash2 /> : <ShieldCheck />}</span>
          <div><p>{destructive ? "High-impact action" : "Human approval required"}</p><h2 id="confirmation-title">{destructive ? "Review deletion" : "Review proposed change"}</h2></div>
          <button className="icon-button" onClick={onClose} aria-label="Close confirmation"><X size={19} /></button>
        </header>

        {result ? (
          <div className="confirmation-result" role="status">
            <CheckCircle2 size={38} />
            <h3>Action {result}</h3>
            <p>The final status was recorded in the audit trail.</p>
            <button className="primary-button" onClick={onClose}>Done</button>
          </div>
        ) : (
          <>
            <div className="action-context">
              <div><span>Spreadsheet</span><strong>Current connection</strong></div>
              <ArrowRight size={16} />
              <div><span>Worksheet</span><strong>{sheet || "Unknown"}</strong></div>
              <ArrowRight size={16} />
              <div><span>Action</span><strong>{action.tool_name || "Spreadsheet update"}</strong></div>
            </div>
            <div className="expiry-note"><Clock3 size={16} /><span>Expires {action.expires_at ? new Date(action.expires_at).toLocaleString() : "soon"} · Approver: current user</span></div>
            <div className="diff-grid">
              <section><h3>Current value / expected state</h3><EntryList value={action.expected_state ?? action.target} /></section>
              <section><h3>Proposed value</h3><EntryList value={action.proposed_change} /></section>
            </div>
            <div className="warning-note"><AlertTriangle size={18} /><p><strong>This uses fresh source data.</strong> If the target changed since staging, execution will stop and require a new proposal.</p></div>
            {destructive && (
              <label className="typed-confirmation">
                <span>Type <strong>DELETE</strong> to enable approval</span>
                <input value={typed} onChange={(event) => setTyped(event.target.value)} autoComplete="off" />
              </label>
            )}
            {error && <div className="inline-alert error" role="alert">{error}</div>}
            <footer>
              <button className="secondary-button danger-text" onClick={onReject} disabled={busy}>Reject action</button>
              <button className={`primary-button ${destructive ? "danger-button" : ""}`} onClick={onConfirm} disabled={busy || (destructive && typed !== "DELETE")}>
                {busy ? "Executing…" : destructive ? "Approve deletion" : "Approve and execute"}
              </button>
            </footer>
          </>
        )}
      </div>
    </div>
  );
}
