const { test, expect } = require("@playwright/test");

test("real backend workspace controls work end to end", async ({ page }) => {
  test.skip(process.env.SHEETSENSE_LIVE_BROWSER !== "1", "Runs only against an explicitly started live backend.");

  const failedApiResponses = [];
  const browserErrors = [];
  page.on("pageerror", error => browserErrors.push(error.message));
  page.on("console", message => {
    if (message.type() === "error") browserErrors.push(message.text());
  });
  page.on("response", response => {
    const url = new URL(response.url());
    if (url.origin === "http://127.0.0.1:8765" && url.pathname !== "/" && response.status() >= 400) {
      failedApiResponses.push(`${response.status()} ${url.pathname}`);
    }
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Connect a workspace" }).click();
  await page.getByLabel("API key").fill("dev-key-123");
  await page.getByRole("button", { name: "Open existing workspace" }).click();

  await expect(page.getByRole("heading", { name: "Live data" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Sheet1/ })).toBeVisible();
  await expect(page.getByText("Live connection")).toBeVisible();

  await page.getByRole("button", { name: "Filter" }).click();
  await expect(page.getByLabel("Column")).toBeVisible();
  await page.keyboard.press("/");
  await expect(page.getByLabel("Search visible rows")).toBeFocused();

  await page.getByRole("button", { name: "Analytics" }).click();
  await expect(page.getByRole("heading", { name: "Build an honest summary" })).toBeVisible();
  await page.getByRole("button", { name: "Schema" }).click();
  await expect(page.getByRole("heading", { name: "Worksheet schema" })).toBeVisible();

  await page.getByRole("button", { name: "Open activity history" }).click();
  await expect(page.getByRole("heading", { name: "Activity history" })).toBeVisible();
  await page.getByRole("button", { name: "Open workspace settings" }).click();
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();

  const assistant = page.getByRole("button", { name: "Assistant" });
  await assistant.click();
  await expect(assistant).toHaveAttribute("aria-expanded", "false");
  await assistant.click();
  await expect(assistant).toHaveAttribute("aria-expanded", "true");

  await page.screenshot({ path: "C:/tmp/sheetsense-live-browser.png", fullPage: true });
  expect(failedApiResponses, `Unexpected API failures: ${failedApiResponses.join(", ")}`).toEqual([]);
  expect(browserErrors, `Unexpected browser errors: ${browserErrors.join(", ")}`).toEqual([]);
});
