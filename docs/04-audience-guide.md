# Audience Guide

Different stakeholders engage with the Flywheel at different layers. Use the section that matches your role.

## For Leadership (CTO, VP Engineering)

- **Why it matters**: v1 targets inference **cost & latency** reduction by 50-98% while maintaining quality; future releases will pursue accuracy and agentic insights.
- **Mental Model**: Treat the flywheel as a *flashlight* that reveals promising smaller models, not an autopilot that swaps models automatically.
- **Expectations & KPIs**:
  - Cost per 1,000 tokens before/after Flywheel cycles
  - Percentage of workloads covered by instrumentation
  - Turn-around time for one Flywheel iteration (**data** → **eval** → **candidate**)
- **Organizational Investments**:
  1. **Data Logging**: green-light adding prompt/completion logs to production.
  2. **GPU/CPU Budgets**: allocate capacity for evaluator + fine-tune jobs (bursty workloads).
  3. **Review Process**: define who signs off on model promotion and what checklists (safety, compliance) apply.
- **Risk Mitigation**: Early cycles may yield *no* winner; that is a success signal that data or techniques must evolve—not a failure of the platform.

## For Product Managers

- **Opportunity**: Iterate on model quality/features without a full research team.
- **Key Questions to Answer**:
  1. Which *workloads* (features, agent nodes) matter most for cost or latency?
  2. What accuracy or UX thresholds are non-negotiable?
- **Your Inputs to Flywheel**:
  - Provide clear *workload IDs* and user intent descriptions (used for eval splitting and future classification).
  - Flag workloads that carry extra compliance or brand-risk sensitivity.
- **Metrics Dashboard** (latency & cost first, accuracy later):
  - Track evaluation scores vs. reference model per workload.
  - Monitor cost deltas for candidate models surfaced by Flywheel.

## For Researchers / ML Engineers

- **What you get**:
  - Auto-generated evaluation datasets (base, ICL, fine-tune) from live traffic.
  - One-click comparative evaluation across many NIMs.
  - Fine-tuning jobs (LoRA) with sensible defaults.
- **How to Drill Deeper**:
  1. Inspect *divergent answers* between reference and candidate models; add them to a specialist evaluation set if needed.
  2. Experiment with advanced data-splitting or per-workload hyper-parameters.
  3. Incorporate **test-time compute** in cost models: `total_tokens × latency`.
- **Caveats & Gotchas**:
  - Flywheel performs *distillation*, not RLHF/DPO.
  - The system does **not** ingest thumbs-up / thumbs-down user feedback; if you want preference-based training, you can extend the pipeline.

## For Application Engineers

- **Instrumentation Checklist**

  | Task | Minimal | Local quick-start (manual) | Production exporter (auto) |
  |------|---------|---------------------------|----------------------------|
  | Log prompt & completion text | ✅ | Provided JSONL sample | ✅ (streamed) |
  | Include `workload_id` | ✅ | Provided JSONL sample | ✅ |
  | Add long-form `description` |  | Optional | Recommended |
  | Record latency, tokens_in/out |  | Optional | Recommended |

- **Implementation Tips**:
  1. Use the provided exporter or send JSON lines directly to Elasticsearch.
  2. Keep log payload sizes reasonable; truncate long passages if not relevant.
- **Debugging Tools**: `./scripts/run-dev.sh` spins up Kibana (browse `log-store-*` index) for real-time ingestion checks, and Flower for task queue status.
- **After Flywheel Runs**: Query the API endpoint `/api/jobs/{id}` or open the example notebook to review results.
