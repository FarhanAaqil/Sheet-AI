const { defineConfig } = require("@playwright/test");
module.exports = defineConfig({
  testDir: "tests/browser",
  timeout: 30000,
  use: { baseURL: "http://127.0.0.1:8765", trace: "retain-on-failure" },
  webServer: {
    command: "python -m uvicorn main:app --host 127.0.0.1 --port 8765",
    url: "http://127.0.0.1:8765/health/live",
    reuseExistingServer: true,
    env: { APP_ENV: "development", ENV: "local_no_redis", ALLOW_INSECURE_NO_AUTH: "true" }
  }
});
