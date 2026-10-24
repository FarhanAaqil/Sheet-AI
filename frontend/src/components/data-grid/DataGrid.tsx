import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Columns3,
  Filter,
  RefreshCw,
  Search,
  SlidersHorizontal
} from "lucide-react";
import { useMemo, useState } from "react";
import type { SheetData } from "../../api/types";

type Props = {
  result?: SheetData;
  loading: boolean;
  error?: string;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onRefresh: () => void;
};

function formatCell(value: unknown) {
  if (value === null || value === undefined || value === "") return <span className="empty-cell">Empty</span>;
  if (typeof value === "number") return new Intl.NumberFormat().format(value);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

export function DataGrid({ result, loading, error, page, pageSize, onPageChange, onRefresh }: Props) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ column: string; direction: "asc" | "desc" }>();
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [columnsOpen, setColumnsOpen] = useState(false);

  const columns = (result?.columns ?? []).filter((column) => !hidden.has(column));
  const rows = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    let next = [...(result?.data ?? [])];
    if (normalized) {
      next = next.filter((row) =>
        Object.values(row).some((value) => String(value ?? "").toLocaleLowerCase().includes(normalized))
      );
    }
    if (sort) {
      next.sort((left, right) => {
        const a = left[sort.column];
        const b = right[sort.column];
        const comparison = String(a ?? "").localeCompare(String(b ?? ""), undefined, {
          numeric: true,
          sensitivity: "base"
        });
        return sort.direction === "asc" ? comparison : -comparison;
      });
    }
    return next;
  }, [query, result?.data, sort]);

  const toggleSort = (column: string) => {
    setSort((current) =>
      current?.column === column
        ? { column, direction: current.direction === "asc" ? "desc" : "asc" }
        : { column, direction: "asc" }
    );
  };

  if (error) {
    return (
      <section className="state-panel" role="alert">
        <div className="state-illustration error">!</div>
        <h2>We couldn’t load this worksheet</h2>
        <p>{error}</p>
        <button className="secondary-button" onClick={onRefresh}><RefreshCw size={16} /> Try again</button>
      </section>
    );
  }

  return (
    <section className="data-workspace" aria-labelledby="grid-heading">
      <div className="workspace-toolbar">
        <div>
          <p className="section-kicker">Spreadsheet workspace</p>
          <h1 id="grid-heading">Live data</h1>
        </div>
        <div className="toolbar-actions">
          <div className="search-control">
            <Search size={16} aria-hidden="true" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search visible rows"
              aria-label="Search visible rows"
            />
            <kbd>/</kbd>
          </div>
          <button className="secondary-button"><Filter size={16} /> Filter</button>
          <div className="popover-wrap">
            <button className="secondary-button" onClick={() => setColumnsOpen(!columnsOpen)} aria-expanded={columnsOpen}>
              <Columns3 size={16} /> Columns
            </button>
            {columnsOpen && (
              <div className="column-popover">
                <strong>Visible columns</strong>
                {(result?.columns ?? []).map((column) => (
                  <label key={column}>
                    <input
                      type="checkbox"
                      checked={!hidden.has(column)}
                      onChange={() =>
                        setHidden((current) => {
                          const next = new Set(current);
                          next.has(column) ? next.delete(column) : next.add(column);
                          return next;
                        })
                      }
                    />
                    <span>{column}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
          <button className="icon-button bordered" onClick={onRefresh} aria-label="Refresh live data">
            <RefreshCw size={17} className={loading ? "spin" : ""} />
          </button>
        </div>
      </div>

      <div className="freshness-row">
        <span className="live-indicator"><span /> Live source</span>
        <span>
          {result?.last_refreshed_at
            ? `Refreshed ${new Date(result.last_refreshed_at).toLocaleString()}`
            : "Freshness is reported after the first successful load"}
        </span>
        {typeof result?.cache_age_seconds === "number" && <span>{Math.round(result.cache_age_seconds)}s cache age</span>}
      </div>

      <div className="table-card" aria-busy={loading}>
        {loading ? (
          <div className="table-skeleton" aria-label="Loading worksheet">
            {Array.from({ length: 9 }).map((_, index) => <span key={index} />)}
          </div>
        ) : rows.length === 0 ? (
          <div className="empty-table">
            <SlidersHorizontal size={28} />
            <h2>No matching rows</h2>
            <p>Adjust your search or filters to show more data.</p>
          </div>
        ) : (
          <div className="table-scroller" role="region" aria-label="Worksheet data" tabIndex={0}>
            <table>
              <thead>
                <tr>
                  <th className="row-number">#</th>
                  {columns.map((column) => (
                    <th key={column}>
                      <button onClick={() => toggleSort(column)}>
                        {column}
                        {sort?.column === column &&
                          (sort.direction === "asc" ? <ArrowUp size={13} /> : <ArrowDown size={13} />)}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    <th className="row-number" scope="row">{page * pageSize + rowIndex + 1}</th>
                    {columns.map((column) => <td key={column}>{formatCell(row[column])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <footer className="pagination">
          <span>
            {result ? `${result.offset + 1}–${result.offset + result.returned_rows} of ${result.total_rows}` : "0 rows"}
          </span>
          <div>
            <button className="icon-button bordered" disabled={page === 0} onClick={() => onPageChange(page - 1)} aria-label="Previous page">
              <ArrowLeft size={16} />
            </button>
            <span>Page {page + 1}</span>
            <button
              className="icon-button bordered"
              disabled={!result?.next_offset}
              onClick={() => onPageChange(page + 1)}
              aria-label="Next page"
            >
              <ArrowRight size={16} />
            </button>
          </div>
        </footer>
      </div>
    </section>
  );
}
