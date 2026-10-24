import { Bot, ChevronDown, Database, Send, Sparkles, User } from "lucide-react";
import { FormEvent, useState } from "react";
import type { PendingAction } from "../../api/types";

export type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  tools?: string[];
  source?: string;
  rowsScanned?: number;
  truncated?: boolean;
};

type Props = {
  sheet?: string;
  freshness?: string;
  messages: Message[];
  busy: boolean;
  error?: string;
  pending?: PendingAction;
  onSend: (message: string) => void;
  onReviewAction: () => void;
};

const suggestions = [
  "Summarize the visible trends",
  "Which rows need attention?",
  "Show missing values by column"
];

export function AssistantPanel({
  sheet,
  freshness,
  messages,
  busy,
  error,
  pending,
  onSend,
  onReviewAction
}: Props) {
  const [draft, setDraft] = useState("");
  const submit = (event: FormEvent) => {
    event.preventDefault();
    const value = draft.trim();
    if (!value || busy) return;
    onSend(value);
    setDraft("");
  };

  return (
    <div className="assistant-inner">
      <header className="assistant-header">
        <div><span className="assistant-mark"><Sparkles size={17} /></span><div><strong>AI assistant</strong><span>Grounded in your workspace</span></div></div>
        <span className="model-badge"><span /> Ready</span>
      </header>
      <div className="assistant-context">
        <Database size={15} />
        <span><strong>{sheet || "No active sheet"}</strong>{freshness ? ` · ${freshness}` : ""}</span>
      </div>

      <div className="message-list" aria-live="polite" aria-busy={busy}>
        {!messages.length && (
          <div className="assistant-welcome">
            <span><Bot size={22} /></span>
            <h2>What would you like to understand?</h2>
            <p>I use the active worksheet and show exactly which tools and sources support the answer.</p>
            <div className="suggestion-list">
              {suggestions.map((suggestion) => (
                <button key={suggestion} onClick={() => onSend(suggestion)} disabled={!sheet}>{suggestion}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((message) => (
          <article key={message.id} className={`message ${message.role}`}>
            <span className="message-avatar" aria-hidden="true">
              {message.role === "assistant" ? <Bot size={15} /> : <User size={15} />}
            </span>
            <div>
              <strong>{message.role === "assistant" ? "SheetSense" : "You"}</strong>
              <p>{message.text}</p>
              {message.role === "assistant" && (message.tools?.length || message.source) && (
                <details className="answer-details">
                  <summary><ChevronDown size={14} /> How this answer was produced</summary>
                  <dl>
                    <div><dt>Source</dt><dd>{message.source || sheet || "Active worksheet"}</dd></div>
                    <div><dt>Tools</dt><dd>{message.tools?.join(", ") || "Direct response"}</dd></div>
                    {message.rowsScanned !== undefined && <div><dt>Rows scanned</dt><dd>{message.rowsScanned}</dd></div>}
                    <div><dt>Truncated</dt><dd>{message.truncated ? "Yes" : "No"}</dd></div>
                  </dl>
                </details>
              )}
            </div>
          </article>
        ))}
        {busy && <div className="thinking" role="status"><i /><i /><i /><span>Analyzing live data…</span></div>}
        {error && <div className="inline-alert error" role="alert">{error}</div>}
        {pending && (
          <div className={`mutation-proposal ${pending.tool_name?.includes("delete") ? "danger" : ""}`}>
            <strong>{pending.tool_name?.includes("delete") ? "Deletion needs approval" : "Change needs approval"}</strong>
            <p>No write has happened. Review the target and exact change before approving.</p>
            <button className="primary-button" onClick={onReviewAction}>Review proposed change</button>
          </div>
        )}
      </div>

      <form className="composer" onSubmit={submit}>
        <label htmlFor="assistant-message" className="sr-only">Ask about this worksheet</label>
        <textarea
          id="assistant-message"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
          placeholder={sheet ? `Ask about ${sheet}…` : "Choose a sheet to begin"}
          disabled={!sheet || busy}
          rows={2}
        />
        <div><span>Enter to send · Shift + Enter for a new line</span><button aria-label="Send message" disabled={!draft.trim() || busy}><Send size={17} /></button></div>
      </form>
    </div>
  );
}
