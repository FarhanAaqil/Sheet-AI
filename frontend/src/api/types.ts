export type Connection = {
  connection_id: string;
  display_name?: string;
  spreadsheet_id?: string;
  permission_level?: string;
  last_sync_at?: string | null;
};

export type SheetData = {
  sheet_name: string;
  connection_id: string;
  total_rows: number;
  columns: string[];
  data: Record<string, unknown>[];
  offset: number;
  limit: number;
  returned_rows: number;
  truncated: boolean;
  next_offset: number | null;
  source: string;
  last_refreshed_at?: string;
  cache_age_seconds?: number;
};

export type PendingAction = {
  action_id: string;
  tool_name?: string;
  target?: Record<string, unknown>;
  proposed_change?: Record<string, unknown>;
  expected_state?: Record<string, unknown>;
  expires_at?: string;
  requires_confirmation?: boolean;
};

export type ChatResponse = {
  answer: string;
  session_id: string;
  connection_id: string;
  tool_calls_made: string[];
  raw_steps?: unknown[];
  pending_action?: PendingAction;
};

export type AuditEvent = {
  id?: number;
  event_type?: string;
  outcome?: string;
  principal_id?: string;
  resource_id?: string;
  request_id?: string;
  trace_id?: string;
  created_at?: string;
  details?: Record<string, unknown>;
};

export type ApiErrorShape = {
  code: string;
  message: string;
  status: number;
  errorId?: string;
};
