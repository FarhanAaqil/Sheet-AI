const { test, expect } = require("@playwright/test");
const AxeBuilder = require("@axe-core/playwright").default;

async function mockWorkspace(page, options = {}) {
  await page.route("**/connections", route => route.fulfill({
    json: { connections: [{ connection_id: "workspace-1", display_name: "Revenue operations", permission_level: "admin" }] }
  }));
  await page.route(/\/sheets(\?.*)?$/, route => route.fulfill({ json: { sheets: ["Orders", "Customers"] } }));
  await page.route("**/sheets/Orders/data?*", route => route.fulfill({
    json: {
      sheet_name: "Orders", connection_id: "workspace-1", total_rows: 2,
      columns: ["Order ID", "Region", "Revenue", "Status"],
      data: [
        { "Order ID": "ORD-1042", Region: "North", Revenue: 18400, Status: "Paid" },
        { "Order ID": "ORD-1043", Region: "West", Revenue: 9200, Status: "Pending" }
      ],
      offset: 0, limit: 100, returned_rows: 2, truncated: false, next_offset: null,
      source: "google_sheets", cache_age_seconds: 4
    }
  }));
  await page.route("**/sessions", route => route.fulfill({ json: { session_id: "session-test" } }));
  await page.route("**/chat", route => route.fulfill({
    json: options.chat ?? {
      answer: "North has the highest visible revenue.",
      session_id: "session-test", connection_id: "workspace-1",
      tool_calls_made: ["aggregate_data"], raw_steps: null, pending_action: null
    }
  }));
  await page.goto("/");
  await page.getByRole("button", { name: "Continue to an unconfigured workspace" }).click();
  await expect(page.getByRole("heading", { name: "Live data" })).toBeVisible();
}

test("onboarding has no serious accessibility violations", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Turn live sheets into clear decisions." })).toBeVisible();
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(results.violations.filter(item => ["serious", "critical"].includes(item.impact))).toEqual([]);
});

test("workspace renders real API data and grounded assistant evidence", async ({ page }) => {
  await mockWorkspace(page);
  await expect(page.getByText("ORD-1042")).toBeVisible();
  await page.getByPlaceholder("Ask about Orders…").fill("Which region leads?");
  await page.getByLabel("Send message").click();
  await expect(page.getByText("North has the highest visible revenue.")).toBeVisible();
  await page.getByText("How this answer was produced").click();
  await expect(page.getByText("aggregate_data")).toBeVisible();
});

test("model content is rendered as inert text", async ({ page }) => {
  const payload = "<img src=x onerror=window.__xss=true> Safe textual result";
  await mockWorkspace(page, {
    chat: {
      answer: payload, session_id: "session-test", connection_id: "workspace-1",
      tool_calls_made: [], raw_steps: null, pending_action: null
    }
  });
  await page.getByPlaceholder("Ask about Orders…").fill("Show result");
  await page.getByLabel("Send message").click();
  await expect(page.getByText(payload)).toBeVisible();
  expect(await page.evaluate(() => window.__xss)).toBeUndefined();
});

test("destructive proposal shows an exact approval diff", async ({ page }) => {
  await mockWorkspace(page, {
    chat: {
      answer: "I staged the deletion; nothing has been written.",
      session_id: "session-test", connection_id: "workspace-1", tool_calls_made: ["delete_row"],
      pending_action: {
        action_id: "action-1", tool_name: "delete_row", requires_confirmation: true,
        target: { "Order ID": "ORD-1043" }, expected_state: { Status: "Pending" },
        proposed_change: { operation: "delete row" }, expires_at: "2030-01-01T00:00:00Z"
      }
    }
  });
  await page.getByPlaceholder("Ask about Orders…").fill("Delete ORD-1043");
  await page.getByLabel("Send message").click();
  await expect(page.getByRole("heading", { name: "Review deletion" })).toBeVisible();
  await expect(page.getByText("delete row")).toBeVisible();
  await expect(page.getByRole("button", { name: "Approve deletion" })).toBeDisabled();
});

test("mobile layout exposes sheet drawer and full-height assistant", async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 800 });
  await mockWorkspace(page);
  await page.getByLabel("Open sheet browser").click();
  await expect(page.getByRole("complementary", { name: "Sheet browser" })).toHaveClass(/open/);
  await page.getByRole("button", { name: "Assistant" }).click();
  await expect(page.getByRole("complementary", { name: "AI assistant" })).not.toHaveClass(/open/);
});

test("liveness returns trace headers", async ({ request }) => {
  const response = await request.get("/health/live", {
    headers: { "X-Request-ID": "browser-request-123", "X-Trace-ID": "browser-trace-123" }
  });
  expect(response.ok()).toBeTruthy();
  expect(response.headers()["x-request-id"]).toBe("browser-request-123");
  expect(response.headers()["x-trace-id"]).toBe("browser-trace-123");
});
