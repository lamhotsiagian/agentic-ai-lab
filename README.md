# agentic-ai-lab

Runnable companion code for **Cracking Agentic AI System Design Interviews** (AI Engineering Career Series).

Every chapter maps to an executable module and an interactive Streamlit lab. The code runs entirely on your local machine using Ollama for language models, with deterministic offline fallback modes for immediate unit testing.

> **Parity Invariant**: Every class and function printed in the book exists in this repository, verified automatically by `scripts/check_book_parity.py`.

---

## Quick Start

```bash
git clone https://github.com/lamhotsiagian/agentic-ai-lab.git
cd agentic-ai-lab

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run any chapter lab
streamlit run chapter_01/app.py
```

### Run Tests & Parity Check

```bash
./scripts/run_tests.sh unit       # 32 unit tests, zero external services required
./scripts/run_tests.sh parity     # assert 100% parity with book listings
./scripts/run_tests.sh all        # run full suite
```

---

## Chapter Map

| Chapter | Module | Lab App | Topic & Architecture |
|---|---|---|---|
| 01 | `chapter_01/bounded_loop.py` | `app.py` | Bounded agent control loop, step budgets, termination reasons |
| 02 | `chapter_02/design_console.py` | `app.py` | 6-component reference architecture, typed state, degradation ladder |
| 03 | `chapter_03/reasoning_router.py` | `app.py` | Reasoning effort classifier, ReWOO plan-and-execute |
| 04 | `chapter_04/durable_graph.py` | `app.py` | Durable graph orchestration, human-in-the-loop approvals |
| 05 | `chapter_05/tool_gateway.py` | `app.py` | Tool gateway, JSON Schema validation, FastMCP incident server |
| 06 | `chapter_06/context_assembler.py` | `app.py` | Intent router, deterministic context assembly & token budgeting |
| 07 | `chapter_07/memory_service.py` | `app.py` | Sensitivity classification, memory write policies, conflict resolution |
| 08 | `chapter_08/skill_library.py` | `app.py` | Dynamic exemplar selection, skill promotion gate |
| 09 | `chapter_09/supervisor.py` | `app.py` | Multi-agent supervisor, handoff envelopes, cycle detection |
| 10 | `chapter_10/protocols.py` | `app.py` | Hardened MCP client with server pinning, A2A delegation |
| 11 | `chapter_11/agent_console.py` | `app.py` | Event-driven agent console, live thought streaming & intervention |
| 12 | `chapter_12/eval_harness.py` | `app.py` | Trajectory assertions, calibrated pairwise LLM-as-a-judge |
| 13 | `chapter_13/telemetry.py` | `app.py` | OpenTelemetry spans, price attribution, tail sampling |
| 14 | `chapter_14/cost_lab.py` | `app.py` | Semantic answer cache, cost governors, speculative routing |
| 15 | `chapter_15/improvement_loop.py` | `app.py` | Production trace failure triage, regression test generation |
| 16 | `chapter_16/defenses.py` | `app.py` | Egress firewall, lethal trifecta guard, provenance tagging |
| 17 | `chapter_17/responsible_ai.py` | `app.py` | Trajectory fairness evaluator, disparate impact intervals |
| 18 | `chapter_18/governance.py` | `app.py` | Escalation router, review SLA queues, agent authority matrix |
| 19 | `chapter_19/computer_use.py` | `app.py` | OS/GUI computer use, visual grounding, step verification |
| 20 | `chapter_20/edge_agent.py` | `app.py` | Local-first routing, device context signals, edge fallbacks |
| 21 | `chapter_21/coding_agent.py` | `app.py` | AST repository localiser, patch integrity guard, anti-cheating |
| 22 | `chapter_22/agentic_retrieval.py` | `app.py` | Adaptive retrieval policy, document grading, query reformulation |
| 23 | `chapter_23/patterns.py` | `app.py` | Circuit breaker, outbox, leaky bucket, saga patterns |
| 24 | `chapter_24/scale_lab.py` | `app.py` | Priority-aware admission control, token bucket limiter |
| 25 | `chapter_25/case_studies.py` | `app.py` | Production case study simulations & telemetry analysis |
| 26 | *Roadmap* | *Guide* | Interview readiness rubric, archetypes, 90-day preparation roadmap |
| 27 | `chapter_27/drills.py` | `app.py` | Coding drills: typed retry, RRF, stream parsing, no-progress detector |
| 28 | `chapter_28/design_drills.py` | `app.py` | Whiteboard system design 45-minute drill simulator |
| 29 | `chapter_29/story_bank.py` | `app.py` | STAR-method behavioral story bank & defense evaluator |
| 30 | `agentops/core/state.py` | `agentops/app.py` | Capstone: Studio typed state machine & multi-agent graph |
| 31 | `agentops/ingest/pipeline.py` | `agentops/app.py` | Content-addressed ingestion pipeline & promotion release gate |
| 32 | `agentops/api/authz.py` | `deploy/app.py` | Principal authorization gateway, incident drills & deployment |

---

## Shared Framework

The `shared/` package powers cross-cutting capabilities across all chapters:
- `shared/config.py`: Environment configurations and model parameters.
- `shared/models.py`: Local model client (Ollama) + offline deterministic fallback.
- `shared/telemetry.py`: Lightweight OpenTelemetry-compatible tracing.
- `shared/db_utils.py`: Multi-tenant isolated vector and relational stores.
- `shared/streamlit_utils.py`: Standardized UI headers, tenant selectors, and metric cards.

---

## Multi-Tenancy & Security

Every lab includes strict tenant partition isolation in the sidebar. Data and vector lookups are scoped to the active tenant ID, preventing cross-tenant leakage across all memory, retrieval, and storage subsystems.
# agentic-ai-lab
# agentic-ai-lab
