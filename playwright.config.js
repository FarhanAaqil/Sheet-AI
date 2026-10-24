const { defineConfig } = require("@playwright/test");
module.exports = defineConfig({
  testDir: "tests/browser",
  timeout: 30000,
  use: { baseURL: "http://127.0.0.1:8765", trace: "retain-on-failure", screenshot: "only-on-failure" },
  projects: [
    { name: "chromium-1440", use: { browserName: "chromium", viewport: { width: 1440, height: 900 } } },
    { name: "firefox-1024", use: { browserName: "firefox", viewport: { width: 1024, height: 768 } } },
    { name: "webkit-768", use: { browserName: "webkit", viewport: { width: 768, height: 900 } } },
    { name: "chromium-mobile", use: { browserName: "chromium", viewport: { width: 360, height: 800 }, isMobile: true } }
  ],
  webServer: {
    command: "python -m uvicorn main:app --host 127.0.0.1 --port 8765",
    url: "http://127.0.0.1:8765/health/live",
    reuseExistingServer: true,
    env: { APP_ENV: "development", ENV: "local_no_redis", ALLOW_INSECURE_NO_AUTH: "true" }
  }
});
