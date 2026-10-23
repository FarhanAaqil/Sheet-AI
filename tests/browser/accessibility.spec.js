const { test, expect } = require("@playwright/test");
const AxeBuilder = require("@axe-core/playwright").default;
test("dashboard has no serious accessibility violations", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("body")).toBeVisible();
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  const serious = results.violations.filter(
    (item) => item.impact === "serious" || item.impact === "critical"
  );
  expect(serious).toEqual([]);
});
test("liveness returns trace headers", async ({ request }) => {
  const response = await request.get("/health/live", {
    headers: { "X-Request-ID": "browser-request-123", "X-Trace-ID": "browser-trace-123" }
  });
  expect(response.ok()).toBeTruthy();
  expect(response.headers()["x-request-id"]).toBe("browser-request-123");
  expect(response.headers()["x-trace-id"]).toBe("browser-trace-123");
});
