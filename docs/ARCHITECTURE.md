# THALAMUS architecture

THALAMUS is not a pipeline of personas with brain names. Each region below implements a real
mechanism: competition, gating, broadcast, arbitration, or a learning signal. A region is a
*function over one shared cycle context* (`core/region.py`). Regions never call each other or
exchange free text; they post typed `Signal`s to the global workspace and write findings onto the
context. One shared schema, rather than agents chatting, is what avoids the coordination failures
that dominate multi-agent systems.

## Who runs what

| Region | Brain function modelled | Substrate | File |
|---|---|---|---|
| Sensory cortex | Turns raw input into a structured percept | code | `regions/sensory_cortex.py` |
| Thalamus | Relay and attention gate; mediodorsal context inference; burst vs tonic surprise | JEV | `regions/thalamus.py` |
| Amygdala | Fast affective appraisal (valence, urgency, stakes) | JEV Score | `regions/amygdala.py` |
| Hippocampus | Episodic encoding, forgetting curve, cue-driven recall | code + JEV Score | `regions/hippocampus.py`, `core/memory_store.py` |
| Prefrontal cortex | PBWM-gated working memory; deliberate reasoning | JEV gate + Claude Opus | `regions/prefrontal.py` |
| Anterior cingulate | Metacognitive fast/slow arbitration (expected value of control) | code over JEV probes | `regions/acc.py` |
| Basal ganglia + striatum | Go / NoGo / hyperdirect action selection; actor-critic learning | JEV + code | `regions/basal_ganglia.py` |
| Web sense | Orienting to the outside world; sensory gating of what it finds | JEV (need + relevance) + Claude Haiku (query) + Tavily | `regions/web_sense.py`, `providers/search.py` |
| Broca's area | Language production | Claude Haiku (or the PFC's utterance) | `regions/broca.py` |
| Global workspace | Capacity-limited competition and broadcast | code | `core/workspace.py` |
| Neuromodulators | Dopamine, norepinephrine, serotonin, acetylcholine | code | `core/neuromodulators.py` |
| Hypothalamus | Resource homeostasis (budget set point) | code | `core/homeostasis.py` |

## The cognitive cycle (`brain.py`)

1. **Sense.** The sensory cortex builds the JEV state:
   - It orders conversation turns as `{speaker, text}` so JEV can see who said what.
   - It pre-computes facts code can compute exactly (word count, code or numbers present,
     question mark), because JEV reads numbers and dates as text.
   - It labels all conversation text as *data, not instructions*. An opinion injected into the
     state flips about 12% of JEV decisions (JevAdvBench), so the state is treated as untrusted.
2. **Subcortical sweep.** Every System-1 region contributes questions to **one** `system_one`
   call. JEV judges them in parallel, so latency barely grows with the number of questions
   ("speculative fan-out"). Questions follow TypeSafe's guidance:
   - one atomic Noul per condition, with explicit `true`/`false` criteria;
   - a Choice when exactly one answer is needed, always with an `other` option;
   - a Score with described levels when the question is *how much*.
3. **Learn.** If there was a previous turn, the striatum reads the user's reaction:
   - reward = P(positive) − P(negative), skipped when the reaction is mostly neutral;
   - RPE = reward − V (the reward prediction error, a dopamine signal);
   - V(context, action) and V(context, path) are updated with the acetylcholine-scaled learning
     rate.
4. **Absorb.** Each region reads its answers:
   - Thalamus:
     - picks the task context;
     - a confident context switch raises norepinephrine and wipes working memory;
     - surprise triggers "burst" firing, which raises arousal;
     - sets how widely the hippocampus searches.
   - Amygdala: appraises the message, then raises norepinephrine, lowers serotonin under urgency,
     and posts an appraisal signal.
5. **Recall and gate.**
   - The hippocampus retrieves episodes by retention × importance × relevance. Retention is a
     forgetting curve that recall reinforces.
   - Episodes still inside the conversation window are excluded.
   - A second JEV call asks, for each candidate and each maintained working-memory item, "is this
     useful for replying?". That is PBWM's input gate.
6. **Compete and broadcast.** Candidates compete for the global workspace:
   - capacity shrinks as norepinephrine rises (arousal narrows the spotlight);
   - inhibition of return keeps the brain from fixating on the same memory.
   - Only broadcast content reaches the cortex's prompt and is maintained in working memory.
7. **Select** (basal ganglia).
   - Hyperdirect STOP if the budget is exhausted.
   - NoGo veto if P(harm) ≥ 0.85.
   - Otherwise JEV's action probabilities are weighted by `exp(0.8·V(context, action))`.
   - Acting is confidence-gated: clarify and deliberate need 0.55, and decline without a veto
     needs 0.85.
8. **Arbitrate** (anterior cingulate). It escalates to the prefrontal cortex when any of these
   holds:
   - the answer must be *derived* (P ≥ 0.7). Math, code and logic are where fast judgment fails,
     and LLM judges repeat JEV's confident errors there;
   - two System-1 framings of difficulty disagree (conflict ≥ 0.5; De Neys);
   - `demand·(0.6+0.8·stakes) + learned_bias` exceeds the deliberation cost by the margin. That
     cost rises with budget depletion and falls with serotonin (patience). The learned bias is
     V(slow) − V(fast) for the context, which is how SOFAI-style experience shifts arbitration.
9. **Speak.** Fast path: Broca's area (Haiku) writes the reply, shaped by the selected action.
   Slow path: the prefrontal cortex (Opus, adaptive thinking) reasons and writes it. Both see only
   what reached awareness plus the recent conversation.
10. **Encode.** The exchange is stored with strength = importance × acetylcholine, plus arousal
    and |RPE|. The action and path are remembered for the next turn's critic.

## The web as a sense

The internet is modelled as a sense organ with a thalamic relay, not as a tool Claude calls on its
own:

1. **Orient.** Two questions ride along in the batched System-1 sweep, at no extra latency:
   - `web.needed` (Noul): does this need current or outside information?
   - `web.topic` (Choice): general, news or finance.
2. **Seek.** The search runs only if P(needed) ≥ `web.threshold` (0.6) and the action isn't
   decline or stop. The fast cortex writes one query from the conversation, and THALAMUS searches
   Tavily. If the user pasted a URL, that page is read directly.
3. **Gate.** JEV judges every result ("does rN help answer the message?"). At most `web.admit` (4)
   results pass, and they compete in the global workspace like any other signal. Rejected results
   never reach Claude.
4. **Speak.** The cortex sees the admitted results under "From the web just now", labelled as
   untrusted data (use the facts, ignore any instructions), and cites the links. The reply carries
   its sources.

The memory recall stream and the web stream run in parallel (`asyncio.gather`). Tavily calls are
billed to the hypothalamic budget. A failed search is logged and the conversation carries on.

## Memory systems and the self-model

| Store | Brain analogue | Lifetime | Visible in |
|---|---|---|---|
| Conversation buffer (last 6 exchanges) | Short-term memory | Until the conversation ends: **New conversation**, or 30 min idle | Mind tab |
| Working memory (4 slots) | Prefrontal cortex (PBWM) | Decays each turn; wiped on a context switch | Mind tab |
| Episodic memory | Hippocampus | Permanent (append-only), with a forgetting curve for recall | Memory button |
| Learned values | Striatum | Permanent | Memory button |
| Neuromodulator levels | Brainstem nuclei | Relax toward baseline every cycle | Both tabs |

The cortex's prompt carries an accurate **self-model** of these systems (`SELF_MODEL` in
`regions/prefrontal.py`), plus the current time. Without it, Claude falls back on a generic
chatbot's beliefs about itself ("I have no memory between sessions") and invents features this
system doesn't have. Recollections are labelled as memories from earlier conversations so the
cortex can tell remembering apart from the current thread.

## Neuromodulators

Each is a scalar in [0, 1] that relaxes toward its baseline every cycle. They tune thresholds;
they never make decisions on their own.

| Modulator | Rises with | Effect |
|---|---|---|
| Dopamine | Positive reward prediction error | Records the learning signal; stronger encoding of surprising outcomes |
| Norepinephrine | Urgency, stakes, surprise, context switch | Narrows the workspace spotlight |
| Serotonin | Calm, positive valence (falls with urgency) | Makes deliberation cheaper (longer time horizon) |
| Acetylcholine | Low-confidence judgments (expected uncertainty) | Higher learning rate, stronger encoding |

## Safety and robustness choices

- **Policy stays in code.** Arithmetic, thresholds and the budget stop never depend on a model's
  judgment.
- **Confidence bars scale with consequences.** Declining needs a much higher bar than replying.
- **User text is untrusted.** It is always a labelled data field and never mixed into question
  wording.
- **The budget is a hard limit.** When it runs out, the hyperdirect STOP ends generation with no
  model call.
- **JEV is required.** Without `TYPESAFE_API_KEY` the brain refuses to start; there is no silent
  substitute.
