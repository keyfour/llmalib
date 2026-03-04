# research — Arxiv Findings That Informed the Design

This document records the specific research findings that shaped `llmalib`'s architecture
and implementation decisions. Every design choice in the library can be traced to at
least one empirical result cited here.

---

## 1. Small Models Excel at Narrow Tasks

**Source:** *Small Language Models are the Future of Agentic AI* (arxiv:2506.02153)

**Finding:** SLMs are 10–30× cheaper in latency, energy, and FLOPs than 70B+ models.
Their reliability in agentic workflows improves dramatically when each invocation is
given a focused, narrow task rather than a broad instruction. Each invocation is also
a natural source of data for future specialisation via fine-tuning.

**Design consequence:**
- `Task` is the primary abstraction, not "agent". A task is inherently narrow.
- `Decomposer` breaks large prompts into multiple focused tasks rather than passing
  them whole to a single call.
- The `token_budget` field on `Task` enforces narrowness structurally.

---

## 2. Prompt Optimisation: Instructions + Demonstrations Work Together

**Source:** *Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies*
(arxiv:2502.02533); *AutoPDL: Automatic Prompt Optimization for LLM Agents*
(arxiv:2504.04365)

**Finding:** The most effective prompt optimisation strategy jointly optimises instructions
(the system prompt) and few-shot demonstrations. Demonstrations bootstrapped from the
model's own correct predictions on a validation set are particularly effective.
AutoPDL showed accuracy gains of up to 67.5 percentage points on 3B–70B models by
selecting the right prompting pattern per model.

**Design consequence:**
- `Task.examples` supports inline few-shot examples that are included in the system prompt.
- `memory/examples.py` supports dynamic example retrieval from a store, enabling
  "bootstrap" behaviour: successful runs populate the store, improving future runs.
- `format_schema_hint()` in `validator.py` generates a compact output schema description
  that is injected into every system prompt — a simple but high-impact prompting pattern.

---

## 3. Context Degradation: The Effective Window Is Smaller Than the Technical Limit

**Sources:**
- *Context Rot* (Chroma Research, citing arxiv:2509.21361)
- *Context Length Alone Hurts LLM Performance Despite Perfect Retrieval* (arxiv:2510.05381)
- *The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization*
  (arxiv:2508.21433)

**Findings:**
- LLM performance on real tasks (not simple retrieval benchmarks) degrades as context grows,
  even when the model technically has room.
- The "lost in the middle" effect means models reliably ignore information in the middle of
  long contexts, regardless of its relevance.
- Simple observation masking (truncation) is competitive with LLM-based summarisation for
  context management, and is cheaper (no extra model call, 52% cost reduction in one study).
- Even with perfect retrieval, longer contexts hurt reasoning quality.

**Design consequence:**
- `Task.token_budget` is a mandatory parameter with a sensible default (2048 tokens).
- `context_window.trim_to_budget()` uses truncation (not summarisation) as the default
  strategy. It preserves the system prompt and final user turn, and trims middle history.
- The trim strategy prioritises the most "middle" content first — the content most likely
  to be ignored by the model anyway.
- `BudgetExceededError` is raised (not warned) when the irreducible minimum exceeds budget,
  forcing the developer to fix the configuration rather than silently running with degraded quality.

---

## 4. Hallucination: Structure Constrains, Reflection Corrects

**Sources:**
- *Large Language Models Hallucination: A Comprehensive Survey* (arxiv:2510.06265)
- *Hallucination Detection and Mitigation in Large Language Models* (arxiv:2601.09929)
- *Theoretical Foundations and Mitigation of Hallucination in LLMs* (arxiv:2507.22915)
- *Reducing hallucination in structured outputs via RAG* (arxiv:2404.08189)

**Findings:**
- Hallucination is formally proven to be unavoidable in the general case (arxiv:2401.11817).
  The practical implication: build for *mitigation*, not *elimination*.
- Instruction-based prompts with structured output constraints significantly reduce
  hallucinations compared to open-ended generation.
- Self-consistency checking (comparing multiple sampled outputs for agreement) is effective
  but expensive — requires 3–5 model calls per result.
- Token-level uncertainty is a useful signal but unreliable: models sometimes assert wrong
  facts with high confidence.
- Contextual fact-checking (comparing output against provided input) is tractable without
  external tools.

**Design consequence:**
- Every `Task` requires an `output_schema`. There is no unstructured task.
- `validator.py` uses three-strategy JSON extraction to handle model non-compliance with
  format instructions (a common failure mode for small models).
- `guards.no_content_from_outside_context()` implements lightweight grounding via Jaccard
  similarity — no embeddings, no extra model call.
- The reflection loop in `retry.py` implements iterative refinement with specific, actionable
  error messages rather than generic retries.

---

## 5. Reflection and Retry: Verbal Reinforcement

**Source:** *Reflexion: Language Agents with Verbal Reinforcement Learning*
(Shinn et al., NeurIPS 2024)

**Finding:** Feeding a model its own error as a new conversational turn — verbal
reinforcement — is significantly more effective than resampling with the same prompt.
The model contrasts its new response against the recorded error in its context.

**Design consequence:**
- `retry.build_reflection_message()` constructs a specific error turn, not a generic
  "try again" message.
- The reflection message lists each error individually and explicitly tells the model
  what to fix.
- The full message history (including prior assistant turns and reflection messages) is
  preserved across retry attempts so the model has full context for correction.
- Maximum retries default to 3 — research suggests diminishing returns beyond 3 attempts;
  the model either corrects on the second or third try, or has a fundamental comprehension
  problem with the task.

---

## 6. Plan-then-Execute: Separate Planning from Execution

**Source:** *Architecting Resilient LLM Agents* (arxiv:2509.08646)

**Finding:** Producing a complete step-by-step plan before taking action leads to higher
reasoning quality and more successful task completion. The planner can be a larger/smarter
model; the executor handles narrow steps and can be much simpler. The expensive planning
call is made once; executor calls are cheap and focused.

**Design consequence:**
- `decomposer.py` implements the planner role: one LLM call produces a `DecompositionPlan`.
- `pipeline.py` implements the executor role: each `Task` is a focused executor call.
- The two can use different models — `Decomposer` can be configured to call a larger model
  while tasks run on a smaller, faster local model.

---

## 7. Few-Shot Example Selection and Ordering

**Source:** *Promptomatix: An Automatic Prompt Optimization* (arxiv:2507.14241)

**Findings:**
- Models exhibit strong recency bias: the last few-shot example has 2–3× more influence
  than the first.
- Label consistency across examples is critical — inconsistent formatting reduces few-shot
  effectiveness.
- Edge cases teach boundary recognition but can cause decision paralysis if overrepresented.

**Design consequence:**
- `examples.select_examples()` orders examples by BM25 relevance, placing the most similar
  example last (maximum recency effect).
- All examples are formatted with the same template. Mixing formats is not supported.
- `max_examples=4` by default. Allowing unlimited examples would encourage
  overrepresentation of edge cases and consume token budget.

---

## 8. Memory Inflation and Selective Retention

**Source:** *Memory Management and Contextual Consistency for Long-Running Agents*
(arxiv:2509.25250)

**Finding:** Agents that accumulate all prior history exhibit "self-degradation" — 
performance declines over time as flawed or irrelevant memories pollute the context.
The solution is selective retention with an "intelligent decay" mechanism that keeps
only high-quality, task-relevant results.

**Design consequence:**
- `store.py` supports a `score` field on each entry, allowing callers to mark low-quality
  results (e.g., from tasks that succeeded on attempt 3) with lower scores.
- `store.get_relevant()` ranks entries by BM25 similarity × entry score, naturally
  demoting low-quality results.
- Optional TTL on store entries prevents indefinite accumulation of stale data.
- The pipeline does **not** automatically accumulate all results into the store. Callers
  explicitly decide what to persist.

---

## Summary Table

| Design Decision | Research Source |
|---|---|
| Task as primary abstraction, narrow focus | arxiv:2506.02153 |
| Decomposer (plan-then-execute) | arxiv:2509.08646 |
| Mandatory output schema | arxiv:2510.06265 |
| Token budget per task | arxiv:2509.21361, arxiv:2510.05381 |
| Truncation over LLM summarisation | arxiv:2508.21433 |
| Reflection-based retry with specific errors | Shinn et al. NeurIPS 2024 |
| Schema hint in every system prompt | arxiv:2504.04365 |
| Few-shot examples at end of prompt | arxiv:2507.14241 |
| BM25 for example selection | arxiv:2507.14241 |
| Jaccard grounding guard | arxiv:2510.06265, arxiv:2601.09929 |
| Selective memory with scoring + TTL | arxiv:2509.25250 |
| Max 3 retries | Shinn et al.; diminishing returns empirical finding |
