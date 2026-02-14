## ML Training Loop Rules

### Sequential dispatch (this loop is inherently serial):
- Design/modify architecture → Train → Evaluate → Decide next step
- Each step depends on the previous output

### Parallel dispatch (within training step):
- Hyperparameter sweeps across independent configs
- Training multiple model variants simultaneously

### Scraping
- HLTV blocks regular Playwright scripts via Cloudflare
- For scraping, use the MCP Playwright browser (`mcp__plugin_playwright_playwright__*` tools) — it works and bypasses Cloudflare

### Progress Tracking
- Always update `progress.md` before and after each iteration
- Log all metrics to `results/metrics.json`
- Commit after each successful training run
