---
description: Run iterative model training loop
argument-hint: "<model_description and target metric>"
---

## Mission
Iteratively build, train, and improve an ML model until target metrics are met.

## Process
1. Read `progress.md` for current state
2. If no model exists: design initial architecture based on the task
3. Train using the model-trainer subagent
4. Evaluate results against target: $ARGUMENTS
5. If target not met: analyze failure modes, propose changes, update architecture
6. Log everything to `results/` directory
7. Update `progress.md` with current iteration results
8. Repeat from step 3

## Key Principles
- Start simple, add complexity incrementally
- Never discard a working baseline without saving it
- Track every experiment in `results/experiments.json`
- If stuck for 3+ iterations, try a fundamentally different approach
