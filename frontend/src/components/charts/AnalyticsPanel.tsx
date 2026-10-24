import { BarChart3, TableProperties } from "lucide-react";
import { useMemo, useState } from "react";
import type { SheetData } from "../../api/types";

type Props = { result?: SheetData };

export function AnalyticsPanel({ result }: Props) {
  const columns = result?.columns ?? [];
  const [dimension, setDimension] = useState(columns[0] ?? "");
  const [measure, setMeasure] = useState(columns[1] ?? "");
  const [aggregation, setAggregation] = useState<"count" | "sum" | "average">("count");

  const values = useMemo(() => {
    const groups = new Map<string, number[]>();
    for (const row of result?.data ?? []) {
      const key = String(row[dimension] ?? "Empty");
      const numeric = Number(row[measure]);
      const existing = groups.get(key) ?? [];
      if (Number.isFinite(numeric)) existing.push(numeric);
      groups.set(key, existing);
    }
    return [...groups.entries()].slice(0, 10).map(([label, numbers]) => ({
      label,
      value:
        aggregation === "count"
          ? numbers.length
          : aggregation === "sum"
            ? numbers.reduce((sum, value) => sum + value, 0)
            : numbers.length
              ? numbers.reduce((sum, value) => sum + value, 0) / numbers.length
              : 0
    }));
  }, [aggregation, dimension, measure, result?.data]);
  const max = Math.max(1, ...values.map((item) => item.value));

  return (
    <section className="analytics-workspace">
      <div className="workspace-toolbar">
        <div><p className="section-kicker">Visual analytics</p><h1>Build an honest summary</h1></div>
      </div>
      <div className="analytics-controls">
        <label>Dimension<select value={dimension} onChange={(e) => setDimension(e.target.value)}>{columns.map(c => <option key={c}>{c}</option>)}</select></label>
        <label>Measure<select value={measure} onChange={(e) => setMeasure(e.target.value)}>{columns.map(c => <option key={c}>{c}</option>)}</select></label>
        <label>Aggregation<select value={aggregation} onChange={(e) => setAggregation(e.target.value as typeof aggregation)}><option value="count">Count</option><option value="sum">Sum</option><option value="average">Average</option></select></label>
      </div>
      <div className="chart-card">
        <header><div><BarChart3 size={18} /><strong>{aggregation} of {measure} by {dimension}</strong></div><span>Visible page only</span></header>
        {values.length ? (
          <div className="bar-chart" role="img" aria-label={`${aggregation} of ${measure} grouped by ${dimension}`}>
            {values.map((item, index) => (
              <div className="bar-row" key={item.label}>
                <span title={item.label}>{item.label}</span>
                <div><i style={{ width: `${Math.max(3, item.value / max * 100)}%` }} data-index={index % 5} /></div>
                <strong>{item.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong>
              </div>
            ))}
          </div>
        ) : <div className="empty-table"><BarChart3 size={28} /><h2>No numeric values</h2><p>Choose a measure containing numbers.</p></div>}
      </div>
      <details className="table-alternative">
        <summary><TableProperties size={16} /> Accessible table alternative</summary>
        <table><thead><tr><th>{dimension}</th><th>{aggregation} of {measure}</th></tr></thead><tbody>{values.map(item => <tr key={item.label}><td>{item.label}</td><td>{item.value}</td></tr>)}</tbody></table>
      </details>
    </section>
  );
}
