import { defineConfig, devices } from '@playwright/test';

const PORT = 5510;
const baseURL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: {
    timeout: 10_000
  },
  fullyParallel: true,
  reporter: [['list']],
  use: {
    baseURL,
    trace: 'on-first-retry',
    viewport: { width: 1280, height: 900 }
  },
  webServer: {
    command: `VITE_E2E_TEST=1 VITE_OCR_API_URL=${baseURL}/__mock_ocr npx vite --host 127.0.0.1 --port ${PORT}`,
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    }
  ]
});
