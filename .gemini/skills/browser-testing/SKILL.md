---
name: browser-testing
description: Guidelines and workflow for performing End-to-End (E2E) testing and Visual QA on a frontend application (Next.js/React) after development is complete. Use this skill when the user asks to "test the app in the browser", "run E2E tests", or verify the frontend rendering and user flows.
---

# Browser Testing & Visual QA Workflow

This skill ensures that completed frontend development (especially Next.js/React applications) is rigorously tested in a real browser environment to verify user flows, rendering, and interactions. 

## When to Activate

- After frontend component or page development is completed.
- When verifying complex user flows (e.g., Auth, Checkout, Multi-step forms).
- When checking UI rendering, responsiveness, or animations.
- Before considering a feature "done" or merging a PR.

## Core Principles

### 1. The Browser is the Source of Truth
Unit tests (Vitest) and Integration tests (RTL) are not enough. If it doesn't render correctly or the user cannot click the button in a real browser, the feature is broken.

### 2. Focus on User Journeys (E2E)
Test full paths, not just isolated components.
Example: *Landing Page -> Click CTA -> Login via Modal -> Fill out Profile -> Checkout -> View Dashboard*.

### 3. Visual QA & Interaction Check
Pay attention to:
- Layout shifts or broken CSS.
- Responsive design (Mobile vs. Desktop).
- Animations (Framer Motion) running smoothly.
- Loading states (Skeletons/Spinners) and interactive feedback (Toasts).

## E2E Testing Workflow (Playwright)

### Step 1: Ensure Development Environment is Ready
Before running tests, the app must be running locally.
```bash
# Terminal 1: Install dependencies and start the dev server
npm install
npm run dev
# The app is usually available at http://localhost:3000
```

### Step 2: Initialize Playwright (If not installed)
```bash
# Terminal 2: Ensure Playwright is set up
npm init playwright@latest
# Make sure to install browsers: npx playwright install
```

### Step 3: Write E2E Scenarios (Playwright)
Create test files in the `tests/e2e/` (or similar) directory.

**Example Pattern for Next.js App (tests/e2e/auth-flow.spec.ts):**
```typescript
import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test('user can log in and view dashboard', async ({ page }) => {
    // 1. Visit Landing Page
    await page.goto('http://localhost:3000/');
    
    // 2. Interact with CTA
    await page.getByRole('button', { name: 'Start Analysis' }).click();
    
    // 3. Mock Authentication (if needed) & fill forms
    // Note: Use semantic locators!
    await page.getByLabel('Date of Birth').fill('1990-01-01');
    await page.getByRole('button', { name: 'Continue' }).click();

    // 4. Verify successful navigation/state
    await expect(page).toHaveURL(/.*\/dashboard/);
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Your Dashboard');
  });
});
```

### Step 4: Execute the Tests
Always run E2E tests and ensure they pass.
```bash
npx playwright test
# To view the report
npx playwright show-report
# For visual debugging
npx playwright test --ui 
```

## Manual / Autonomous Agent Browser Testing

If an automated E2E script is not written yet, or if doing a quick exploratory test, use the **Browser Subagent** (if available) or instruct the user to verify manually.

### Agent Workflow using Browser Tools:
1. **Command**: Start the Next.js server (`npm run dev`) in the background.
2. **Action**: Instruct the browser subagent to navigate to `http://localhost:3000`.
3. **Task**: Give the subagent a specific user persona and goal (e.g., "Act as a user who wants to buy a report. Go through the funnel until you hit the payment gateway mock.")
4. **Verification**: After the subagent completes the task, check the final DOM or visual output (screenshot) to confirm the UI is rendered flawlessly.

## Common Browser Testing Pitfalls to Avoid

- ❌ **Testing against a mock without verifying the real integration**: Ensure your mock API endpoints (Next.js route handlers) simulate real network latency.
- ❌ **Ignoring mobile viewports**: Set up Playwright to test both desktop and mobile viewports.
- ❌ **Using brittle CSS selectors**: Always use semantic locators (`getByRole`, `getByText`, `getByLabel`) instead of class names (`.submit-btn-wrapper`).
- ❌ **Not cleaning up state**: Ensure tests act independently; do not depend on the browser state from a previous test.

---

**Remember**: Frontend development is only complete when the user can successfully and beautifully navigate the application in a real browser.
