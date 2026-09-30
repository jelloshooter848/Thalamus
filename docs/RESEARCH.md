# Research notes behind THALAMUS

Compiled 2026-09-30.

**How this was gathered.**
- **Sources:** the JEV video transcripts in this repo, the `typesafe-sdk` 0.7.2 source code, and
  web research.
- **Access limits:** several documentation sites were unreachable from the build environment. For
  those, claims come from search summaries and are marked **[unverified]**.
- **Vendor figures:** benchmark numbers are the vendors' own unless stated otherwise.

## 1. JEV (TypeSafe AI's System One model)

### What it is
- **Not a chatbot.** JEV is a non-generative decision model; it never writes text.
- **How you use it.** You give it a *state* (text, a JSON object or an array) and named typed
  questions. It answers all of them in parallel with calibrated probabilities.
- **Name.** "System One" refers to Kahneman's fast, intuitive System 1.
- **Launch.** Released September 15, 2026. The current model is `jev-1.13.0`; `jev-latest` and
  `jev-preview` both point to it.
- **Pricing.** $0.042 per million input tokens. Output tokens are free.
- **Latency.**
  - TypeSafe claims 70–500 ms end to end.
  - Independent measurements:
    - JevBench: p50 of about 60 ms server-side and about 250 ms including network.
    - LiteLLM: p50 of 127 ms.
    - The JEV-as-a-Judge paper: a median of 0.152 s.
- **Question types.**
  - **Noul.** A yes/no question. Returns only P(yes); there is no separate confidence.
  - **Choice.** Picks one of up to 255 labels. Returns per-label probabilities plus a confidence.
    Accuracy is best with about 20 labels or fewer.
  - **Score.** Rates against 2–10 ordered levels. Returns the expected level, the distribution and
    a confidence.
- **Limits.**
  - Context is 64k tokens, of which 32k is for the state plus the longest question.
  - Text only, no streaming.
  - Rate limits of about 1,200 requests per minute. **[unverified]**
- **Python SDK** (verified from the source code):
  - Install `typesafe-sdk`. Clients are `TypeSafeClient` and `AsyncTypeSafeClient`.
  - Call `system_one(state=..., questions={...})`, which sends `POST https://api.typesafe.ai/v1/systemone`.
  - Responses expose `.answers`, `.nouls`, `.choices`, `.scores` and `.usage`.
  - Configured through the environment variables `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` and
    `TYPESAFE_DEFAULT_MODEL`.
  - Retries are built in: on 408, 429 and 5xx, and on connection errors.

### Using it well
From the transcripts and TypeSafe's guidance:
- **Shape the state.**
  - Use an object, not one long string, whenever there are several related pieces.
  - Make relationships explicit: ordered `{speaker, text}` turns beat separate per-speaker lists.
  - Compute dates and differences, and resolve lookups, in code before sending.
  - Leave out anything the question doesn't need. Irrelevant state lowers accuracy and costs
    tokens.
- **Ask good questions.**
  - Use a Choice when exactly one answer is needed, and always include an `other` option.
  - Use separate Nouls when several conditions can all be true.
  - Use a Score, with described levels, to measure *how much*. A Noul probability is not an
    intensity.
  - Split compound questions into atomic ones and combine them in code.
  - Define the criteria: `true`/`false` for a Noul; descriptions, boundaries and examples for
    Choice options; summaries and signals for Score levels.
  - Batch every question that shares a state into one call. Answers stay independent.
- **Act on the answers.**
  - **Intent routing:** JEV judges, code applies the policy.
  - **Confidence-gated routing:** set the threshold per action according to the cost of being
    wrong.
  - **Composite scoring:** re-weight saved judgments in code without asking JEV again.

### Known weaknesses
From TypeSafe's "jev-1.13 jaggedness" page and independent studies:
- **Literal reading.** It answers exactly what was written, including negations.
- **Arithmetic.** It is poor at counting, numbers and dates. Do these in code.
- **Distraction.** It is swayed by distracting or irrelevant state.
- **Prompt injection.** An injected *opinion* in the state flips 12.1% of decisions, and injected
  commands flip about 10% (JevAdvBench, arXiv 2609.31142). Treat state as untrusted.
- **No consistency guarantee.** Answers can contradict each other: a Noul and its negation can
  sum to more than 1.
- **Uneven calibration.** Per-benchmark ECE (calibration error) is 0.168 ("Just Ask Jev", arXiv
  2609.29429). Recalibrating on your own labelled data helps a lot.
- **Correlated errors.** When JEV is confidently wrong, frontier LLM judges tend to make the same
  mistake ("Wrong in the same places", arXiv 2609.29769). Escalating to an LLM cuts cost but
  doesn't catch those errors, so high-stakes actions need checks against reality.
- **Where it works and where it doesn't.** It is strongest when the verdict can be read directly
  off the text, and weakest when the verdict must be *derived* (math, code, logic). A confidence
  threshold frozen in advance for escalating to an LLM matched the LLM's accuracy at about 41% of
  its cost (JEV-as-a-Judge, arXiv 2609.26550).

### Related patterns seen in the wild
- **LangChain harness:** JEV as a model-routing middleware and as a risk gate on tool calls.
- **browser-use `jev-ultrafast`:** JEV picks each browser action from a text table of page
  elements, at about 178 ms per step.
- **`jev-realtime-sdk`:** JEV makes discrete decisions at 2–10 Hz while millisecond reflexes stay
  in code.

## 2. Brain mechanisms adopted

| Mechanism | Source | Where it lives in THALAMUS |
|---|---|---|
| Global workspace: select, then broadcast | Baars; LIDA (Franklin); Global Workspace Agents (arXiv 2604.08206) | `core/workspace.py` |
| Driver/modulator signal split | Sherman & Guillery | `core/signals.py` |
| Reticular nucleus as attention gate; amygdala→TRN salience bias | Crick; Halassa lab | thalamus recall breadth; appraisal salience |
| Mediodorsal thalamus: context inference and switching | Halassa lab (Neuron 2024; Nat. Commun. 2025) | `regions/thalamus.py` |
| Burst vs tonic firing as surprise | Varela, Ahmad et al. 2024 | thalamus surprise → norepinephrine |
| Fast/slow metacognitive arbitration | SOFAI (npj AI 2025); SOFAI-LM (arXiv 2508.17959) | `regions/acc.py` |
| Expected value of control | Shenhav, Botvinick & Cohen 2013 | ACC value vs cost |
| Conflict between intuitions triggers deliberation | De Neys (BBS 2023) | ACC conflict probes |
| Go / NoGo / hyperdirect action selection | Gurney, Prescott & Redgrave | `regions/basal_ganglia.py` |
| Actor-critic learning; dopamine as reward prediction error | Schultz; actor-critic basal ganglia | striatum critic, `vals` table |
| PBWM working-memory gating | O'Reilly & Frank 2006 | `WorkingMemory` in `regions/prefrontal.py` |
| Neuromodulators as meta-parameters | Doya 2002; Aston-Jones & Cohen 2005; Yu & Dayan 2005 | `core/neuromodulators.py` |
| Complementary learning systems; retrieval by recency, importance and relevance | McClelland et al.; Generative Agents (Park et al.) | `core/memory_store.py` |
| Forgetting curve reinforced by recall | Ebbinghaus; MemoryBank (arXiv 2305.10250) | retention in `recall` |
| Interoceptive homeostasis | Lee et al., Nature Machine Intelligence 2026 | `core/homeostasis.py` |

**Pitfalls deliberately avoided.**
- **"Neuro-washing":** brain labels on a linear pipeline.
- **Free-text agent chatter:** the MAST taxonomy traces most multi-agent failures to coordination
  and specification, not model weakness.
- **Uncontrolled slow-model calls.**
- **Memory poisoning:** raw episodes are append-only and keep their provenance.

## 3. Other novel AI systems that could become brain regions

| System | What it is | Access | Best fit in THALAMUS | Status |
|---|---|---|---|---|
| **Laya** (Convai Innovations) | Open, JEV-compatible decision model (421M, ModernBERT); ~33 ms locally | Hugging Face, Apache-2.0 | Local "brainstem" pre-filter *in front of* JEV. Needs temperature recalibration: ships with ECE 0.466, which drops to 0.081 after recalibrating | Roadmap M3 |
| **GLiNER2.5-Decide** (Fastino) | 340M non-generative decision model; runs on CPU; ~167 ms | `pip install gliner2`, Apache-2.0 | Entity extraction for hippocampal encoding; local reflexes | Roadmap M3 |
| **Mercury 2.5** (Inception) | Diffusion LLM, ~440 tok/s measured; JSON and tool support | OpenAI-compatible API, OpenRouter | Fast Broca's area and inner speech | Roadmap M3 |
| **DiffusionGemma 26B-A4B** | Open-weight diffusion LLM | Hugging Face / vLLM | Self-hosted fast Broca | Optional |
| **Letta sleep-time agents** | Background agents that consolidate memory while idle | open source | Sleep and replay consolidation | Roadmap M2 |
| **Graphiti / Zep** | Bi-temporal knowledge graph memory | `pip install graphiti-core` | Episodic "what happened when" | Roadmap M2 |
| **HippoRAG 2** | Hippocampal-index retrieval (knowledge graph + Personalized PageRank) | `pip install hipporag` | Associative pattern completion | Roadmap M2 |
| **A-MEM** | Self-linking Zettelkasten memory | source | Semantic neocortical memory | Candidate |
| **V-JEPA 2 / 2.1** (Meta) | Latent-prediction video world model | Hugging Face / torch.hub | Visual cortex; prediction error as a real surprise signal | Roadmap M4 |
| **DINOv3, SAM 3.1** (Meta) | Self-supervised features; promptable segmentation | open weights | Ventral "what" and dorsal "where" streams | Roadmap M4 |
| **Kyutai Unmute / Moshi** | Streaming speech-to-text and text-to-speech around any LLM | open source | Auditory cortex and speech output | Roadmap M4 |
| **emotion2vec+** | Vocal emotion recognition | open weights | Vocal affect input to the amygdala | Roadmap M4 |
| **pymdp 1.0** | Active-inference POMDP agents (JAX) | pip | Free-energy action and attention selection | Roadmap M4 |
| **MAPIE / TorchCP** | Conformal prediction | pip | Guaranteed-coverage escalation rule for the ACC | Roadmap M3 |
| **Liquid LFM2.5** | Small on-device vision and audio models | open weights | Always-on local sensory relays | Optional |
| **Cosmos 3** (NVIDIA) | Physical world-model "omnimodel" | open weights, heavy GPU | Mental simulation / imagination | Future |
| **HRM / TRM** | Tiny recursive reasoners trained per task | open source | Trained cerebellar skill modules | Future |
| **Genie 3, Gemini Diffusion, Titans / HOPE, Kona EBM** | World simulation; diffusion LLM; test-time memory; energy-based reasoning | no public access | Dreaming; consolidation; plan-constraint veto | Future |

## 4. Roadmap

- **M2: Memory and sleep.**
  - `thalamus sleep` consolidation: prioritized replay (weighted by dopamine and amygdala tags) →
    semantic facts and reflections that link back to the raw episodes (a guard against identity
    drift).
  - Graphiti or HippoRAG graph memory.
  - **Chunking:** repeated successful System-2 solutions are compiled into JEV decision templates
    (a cerebellum and habit system). A forward model demotes a template when its predictions miss.
- **M3: Other AIs.**
  - Mercury 2.5 as the fast Broca.
  - Laya / GLiNER2.5-Decide as a local pre-filter in front of JEV, never replacing it.
  - Conformal escalation in the ACC.
  - A default-mode "idle thought" mode with a conservative JEV evaluator.
  - Sampled Claude audits of confident JEV verdicts to track calibration drift.
- **M4: Senses and embodiment.**
  - Vision: V-JEPA 2.1 surprise signal, DINOv3, SAM 3.1.
  - Hearing: Unmute, emotion2vec+.
  - pymdp active inference for action selection.

## 5. Sources

**JEV**
- [TypeSafe: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [PyPI: typesafe-sdk](https://pypi.org/project/typesafe-sdk/)
- [GitHub: typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python)
- [TypeSafe docs: jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- [JevBench results](https://github.com/fstandhartinger/jevbench/blob/main/RESULTS-v1.2.md)
- [jev-fanout](https://github.com/TheWebDevel/jev-fanout)
- [Pydantic AI: TypeSafe](https://pydantic.dev/docs/ai/models/typesafe/)
- [LiteLLM: TypeSafe pass-through](https://docs.litellm.ai/docs/pass_through/typesafe)
- [browser-use jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
- [LangChain: Building a harness with Jev](https://events.langchain.com/webinar/building-a-harness-with-jev/)

**Papers**
- [Just Ask Jev](https://arxiv.org/abs/2609.29429)
- [JEV-as-a-Judge](https://arxiv.org/abs/2609.26550)
- [Wrong in the same places](https://arxiv.org/abs/2609.29769)
- [Jev and Laya in agents](https://arxiv.org/abs/2609.28940)
- [Global Workspace Agents](https://arxiv.org/abs/2604.08206)
- [CoALA](https://arxiv.org/abs/2309.02427)
- [SOFAI](https://www.nature.com/articles/s44387-025-00027-5)
- [SOFAI-LM](https://arxiv.org/abs/2508.17959)
- [MAP planner](https://www.nature.com/articles/s41467-025-63804-5)
- [Expected value of control](https://www.cell.com/neuron/fulltext/S0896-6273(13)00607-7)
- [Doya 2002](https://pubmed.ncbi.nlm.nih.gov/12371507/)
- [Yu & Dayan 2005](https://www.cell.com/neuron/fulltext/S0896-6273(05)00362-4)
- [Aston-Jones & Cohen](https://pubmed.ncbi.nlm.nih.gov/16022602/)
- [Mediodorsal thalamus review](https://pubmed.ncbi.nlm.nih.gov/38295791)
- [Thalamic bursts as surprise](https://pubmed.ncbi.nlm.nih.gov/38486972)
- [Generative Agents](https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763)
- [MemoryBank](https://arxiv.org/abs/2305.10250)
- [MAST failure taxonomy](https://arxiv.org/pdf/2503.13657)
- [Homeostatic machine intelligence](https://www.nature.com/articles/s42256-026-01296-8)

**Systems**
- [Laya](https://huggingface.co/convaiinnovations/laya)
- [GLiNER2.5-Decide](https://www.marktechpost.com/2026/09/24/fastino-releases-gliner2-5-decide-a-340m-open-weight-decision-model-that-runs-on-cpu/)
- [Mercury](https://docs.inceptionlabs.ai/get-started/models)
- [Letta sleep-time](https://docs.letta.com/guides/agents/architectures/sleeptime/)
- [Graphiti](https://github.com/getzep/graphiti)
- [HippoRAG](https://github.com/osu-nlp-group/hipporag)
- [V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- [Kyutai Unmute](https://github.com/kyutai-labs/unmute)
- [pymdp](https://github.com/infer-actively/pymdp)
- [MAPIE](https://github.com/scikit-learn-contrib/MAPIE)
