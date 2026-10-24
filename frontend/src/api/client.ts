import type {
  ApiErrorShape,
  AuditEvent,
  ChatResponse,
  Connection,
  SheetData
} from "./types";

type ClientOptions = {
  apiKey: string;
  baseUrl?: string;
};

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly errorId?: string;

  constructor(error: ApiErrorShape) {
    super(error.message);
    this.name = "ApiError";
    this.code = error.code;
    this.status = error.status;
    this.errorId = error.errorId;
  }
}

function messageFromDetail(detail: unknown): { code: string; message: string; errorId?: string } {
  if (typeof detail === "string") return { code: "REQUEST_FAILED", message: detail };
  if (detail && typeof detail === "object") {
    const value = detail as Record<string, unknown>;
    return {
      code: String(value.code ?? "REQUEST_FAILED"),
      message: String(value.message ?? "The request could not be completed."),
      errorId: value.error_id ? String(value.error_id) : undefined
    };
  }
  return { code: "REQUEST_FAILED", message: "The request could not be completed." };
}

export function createApiClient({ apiKey, baseUrl = "" }: ClientOptions) {
  async function request<T>(
    path: string,
    init: RequestInit = {},
    signal?: AbortSignal
  ): Promise<T> {
    const response = await fetch(`${baseUrl}${path}`, {
      ...init,
      signal,
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "X-API-Key": apiKey } : {}),
        ...init.headers
      }
    });
    if (!response.ok) {
      let body: Record<string, unknown> = {};
      try {
        body = await response.json();
      } catch {
        body = {};
      }
      const parsed = messageFromDetail(body.detail ?? body);
      throw new ApiError({ ...parsed, status: response.status });
    }
    return response.json() as Promise<T>;
  }

  return {
    health: (signal?: AbortSignal) =>
      request<{ status: string; sheets_connected: boolean; sheets_count: number }>(
        "/health",
        {},
        signal
      ),
    connections: (signal?: AbortSignal) =>
      request<{ connections: Connection[] }>("/connections", {}, signal),
    connect: (sheetUrl: string) =>
      request<{ status: string; connection_id: string; display_name?: string }>(
        "/sheets/connect",
        { method: "POST", body: JSON.stringify({ sheet_url: sheetUrl }) }
      ),
    sheets: (connectionId?: string, signal?: AbortSignal) =>
      request<{ sheets: string[] }>(
        `/sheets${connectionId ? `?connection_id=${encodeURIComponent(connectionId)}` : ""}`,
        {},
        signal
      ),
    data: (
      sheet: string,
      options: { connectionId?: string; offset?: number; limit?: number; refresh?: boolean },
      signal?: AbortSignal
    ) => {
      const params = new URLSearchParams({
        offset: String(options.offset ?? 0),
        limit: String(options.limit ?? 100),
        refresh: String(Boolean(options.refresh))
      });
      if (options.connectionId) params.set("connection_id", options.connectionId);
      return request<SheetData>(
        `/sheets/${encodeURIComponent(sheet)}/data?${params}`,
        {},
        signal
      );
    },
    schema: (sheet: string, connectionId?: string, signal?: AbortSignal) => {
      const suffix = connectionId
        ? `?connection_id=${encodeURIComponent(connectionId)}`
        : "";
      return request<Record<string, unknown>>(
        `/sheets/${encodeURIComponent(sheet)}/schema${suffix}`,
        {},
        signal
      );
    },
    createSession: () => request<{ session_id: string }>("/sessions", { method: "POST" }),
    chat: (payload: {
      message: string;
      session_id?: string;
      connection_id?: string;
      sheet_name?: string;
    }) =>
      request<ChatResponse>("/chat", {
        method: "POST",
        body: JSON.stringify(payload)
      }),
    confirm: (actionId: string) =>
      request<Record<string, unknown>>(`/actions/${encodeURIComponent(actionId)}/confirm`, {
        method: "POST",
        headers: { "Idempotency-Key": `ui:${actionId}` }
      }),
    reject: (actionId: string) =>
      request<{ status: string }>(`/actions/${encodeURIComponent(actionId)}/reject`, {
        method: "POST"
      }),
    audit: (signal?: AbortSignal) =>
      request<{ events: AuditEvent[] }>("/admin/audit?limit=100", {}, signal)
  };
}
