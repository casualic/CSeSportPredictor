## ML Training Loop Rules

### Sequential dispatch (this loop is inherently serial):
- Design/modify architecture → Train → Evaluate → Decide next step
- Each step depends on the previous output

### Parallel dispatch (within training step):
- Hyperparameter sweeps across independent configs
- Training multiple model variants simultaneously

### Progress Tracking
- Always update `progress.md` before and after each iteration
- Log all metrics to `results/metrics.json`
- Commit after each successful training run
