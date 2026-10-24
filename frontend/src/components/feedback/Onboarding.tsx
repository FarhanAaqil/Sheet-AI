import { ArrowRight, Check, KeyRound, Link2, LockKeyhole, Sparkles, Table2 } from "lucide-react";
import { FormEvent, useState } from "react";
import { z } from "zod";

const urlSchema = z.string().url().refine(
  (value) => /^https:\/\/docs\.google\.com\/spreadsheets\/d\//.test(value),
  "Enter a valid Google Sheets URL."
);

type Props = {
  busy: boolean;
  error?: string;
  onConnect: (apiKey: string, sheetUrl: string) => Promise<void>;
  onContinue: (apiKey: string) => void;
};

export function Onboarding({ busy, error, onConnect, onContinue }: Props) {
  const [step, setStep] = useState(0);
  const [apiKey, setApiKey] = useState("");
  const [sheetUrl, setSheetUrl] = useState("");
  const [validation, setValidation] = useState("");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const result = urlSchema.safeParse(sheetUrl);
    if (!result.success) {
      setValidation(result.error.issues[0]?.message ?? "Enter a valid URL.");
      return;
    }
    setValidation("");
    await onConnect(apiKey, sheetUrl);
  };

  return (
    <main className="onboarding">
      <section className="onboarding-brand">
        <div className="onboarding-logo"><Table2 size={25} /></div>
        <span>SheetSense</span>
      </section>
      <section className="onboarding-card" aria-labelledby="welcome-title">
        <div className="eyebrow"><Sparkles size={15} /> Safer spreadsheet intelligence</div>
        <h1 id="welcome-title">Turn live sheets into clear decisions.</h1>
        <p className="lead">
          Explore, analyze, and update Google Sheets with an AI assistant that keeps every write behind human approval.
        </p>

        {step === 0 ? (
          <>
            <div className="capability-grid">
              <article><Table2 /><strong>Explore live data</strong><span>Filter, sort, inspect schema, and track freshness.</span></article>
              <article><Sparkles /><strong>Ask grounded questions</strong><span>See sources, tools, and truncation with every response.</span></article>
              <article><LockKeyhole /><strong>Approve every change</strong><span>Review an exact diff before anything is written.</span></article>
            </div>
            <button className="primary-button wide" onClick={() => setStep(1)}>
              Connect a workspace <ArrowRight size={17} />
            </button>
            <button className="text-button" onClick={() => onContinue("")}>
              Continue to an unconfigured workspace
            </button>
          </>
        ) : (
          <form onSubmit={submit} className="connect-form">
            <div className="step-indicator"><span className="done"><Check size={13} /></span><span className="line" /><span>2</span></div>
            <label>
              <span><KeyRound size={15} /> API key</span>
              <input
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                type="password"
                autoComplete="off"
                placeholder="Your SheetSense access key (default: dev-key-123)"
              />
              <small>Kept only in this browser tab. Use <code>dev-key-123</code> or leave blank.</small>
            </label>
            <label>
              <span><Link2 size={15} /> Google Sheet URL</span>
              <input
                value={sheetUrl}
                onChange={(event) => setSheetUrl(event.target.value)}
                placeholder="https://docs.google.com/spreadsheets/d/…"
                aria-invalid={Boolean(validation)}
              />
              {validation && <small className="field-error">{validation}</small>}
            </label>
            {(error || validation) && <div className="inline-alert error" role="alert">{error || validation}</div>}
            <div className="permission-note">
              <LockKeyhole size={18} />
              <div><strong>Permission-aware by design</strong><span>SheetSense uses your assigned role and never bypasses Google permissions.</span></div>
            </div>
            <button className="primary-button wide" disabled={busy}>
              {busy ? "Testing connection…" : "Test and connect"} <ArrowRight size={17} />
            </button>
            <button className="text-button" type="button" onClick={() => setStep(0)}>Back</button>
          </form>
        )}
      </section>
      <footer>Private by default · Human-approved writes · Traceable activity</footer>
    </main>
  );
}
