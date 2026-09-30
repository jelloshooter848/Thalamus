# THALAMUS

**T**hought **H**ub for **A**ttention, **L**earning, **A**rbitration, **M**emory, and **U**nified **S**ynthesis.

Most agents use sub-agents as separate *specialists*: a coder, a researcher, a critic, each with
its own persona. THALAMUS uses sub-agents as **components of one brain**. Each region does one
cognitive function, the regions share a single typed workspace, and what they produce together is
one mind.

It is a **dual-process** brain:

| | System 1: fast, intuitive, always on | System 2: slow, deliberate, recruited on demand |
|---|---|---|
| Runs on | [JEV](https://typesafe.ai) (TypeSafe's System One model): typed Choice / Score / yes-no judgments with calibrated probabilities in ~0.1–0.4 s | Claude: Opus for the prefrontal cortex, Haiku for fast speech |
| Regions | thalamus, amygdala, hippocampal gating, anterior cingulate probes, basal ganglia | prefrontal cortex, Broca's area |
| Cost | ~$0.00005 per cycle | only when the anterior cingulate decides it's worth it |

Plain code handles everything that is policy, arithmetic or safety reflex, which is how TypeSafe
recommends using JEV: *JEV judges, code decides.*

```
input ─▶ Sensory cortex ─▶ THALAMUS (gate · context · surprise) ─▶ Global Workspace (top-k broadcast)
                 amygdala (affect) ─┘     hippocampus (recall) ─┘            │
                                                                             ▼
               basal ganglia (Go / NoGo / STOP) ─▶ anterior cingulate (fast or slow?)
                      ├─ fast ─▶ Broca's area (Claude Haiku) ──────────────▶ reply
                      └─ slow ─▶ prefrontal cortex (Claude Opus) ─▶ Broca ─▶ reply
   dopamine · norepinephrine · serotonin · acetylcholine + hypothalamic budget tune every threshold
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how each mechanism works, and
[docs/RESEARCH.md](docs/RESEARCH.md) for the research behind the design and the roadmap for adding
other AIs (vision, hearing, graph memory, sleep consolidation).

## Quick start (no terminal needed)

1. **Download** the repo: the green **Code** button → **Download ZIP**, then unzip it. You can also
   clone it.
2. **Double-click the launcher** in the Thalamus folder:
   - **Windows:** `Start-THALAMUS.bat`
   - **Mac:** `Start-THALAMUS.command`. If macOS says it can't be opened, right-click it → **Open**
     → **Open**.
   - **Linux:** run `./Start-THALAMUS.command` from a terminal.
3. **Enter your keys.** THALAMUS opens in your browser and asks for them:
   - **TypeSafe API key** (required): powers JEV.
   - **Anthropic API key**: powers Claude, from console.anthropic.com → Settings → API Keys.
   - **Workspace ID**: only if your Anthropic key says it "works across workspaces". The ID starts
     with `wrkspc_`.

   Click **Test & save**. Keys are checked live, then saved only on your computer, in `.env`.
4. **Chat.** The side panel has two tabs:
   - **Brain activity** shows every cognitive cycle: which regions fired, fast or slow thinking,
     neuromodulator levels, memories that surfaced, and cost.
   - **Mind** shows what THALAMUS is holding right now: the conversation buffer Claude is sent
     word for word, the recollections in working memory and how strong they are, and which
     memories the hippocampus offered for your last message, including the ones JEV turned away.

**Conversations work like a person's.** A conversation starts fresh when you click **New
conversation** or after 30 minutes without a message. After that, THALAMUS reaches earlier
conversations only through its long-term memory. Use the **Memory** button to see or erase that
memory.

**What the launcher does the first time:**
- It finds Python 3.11 or newer.
  - On Windows it installs Python with `winget` if it's missing.
  - Otherwise it opens python.org for you.
- It creates a private environment (`.venv`) and installs THALAMUS.

After that, it starts right away. It only reinstalls when an update changes the dependencies. Keep
the launcher window open while you chat; close it to stop THALAMUS.

**A TypeSafe API key is required.** JEV runs every System-1 region and there is no substitute
model, so THALAMUS won't think without it.

### Advanced: terminal use

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
thalamus web                # the browser app
thalamus chat --trace       # text chat in the terminal
thalamus doctor             # check keys and connectivity
thalamus memory list        # episodes and the striatum's learned values
thalamus memory clear       # forget everything
pytest                      # offline test suite (test-only fakes, no network)
```

Configuration is optional: copy `thalamus.example.toml` to `thalamus.toml` to change models,
thresholds, budget or the memory location.

## What happens when you send a message

1. **Sensory cortex** (code) builds a structured, pre-computed percept. User text goes in labelled
   fields that are marked as untrusted data.
2. **One batched JEV call** (about 13 questions answered in parallel) serves every subcortical
   region:
   - The **thalamus** infers the task context and detects surprise.
   - The **amygdala** scores valence, urgency and stakes.
   - The **hippocampus** scores how important the message is to remember.
   - The **anterior cingulate** probes how hard the reply is.
   - The **basal ganglia** propose an action and check the harm veto.
   - The **striatum** reads your reaction to the previous reply.
3. **Learning**: the striatum turns your reaction into a reward prediction error, a dopamine
   signal. It updates the learned value of the action and of the fast or slow path the brain
   took last turn.
4. **Memory**:
   - The hippocampus recalls episodes from past sessions.
   - JEV gates the relevant ones into working memory.
   - Everything competes for a limited global workspace, whose capacity shrinks under high arousal.
5. **Action and arbitration**:
   - The basal ganglia pick an action: respond, clarify, deliberate, decline or stop.
   - The anterior cingulate decides fast versus slow. It weighs how demanding the reply is, what
     is at stake, and whether JEV's probes contradict each other, against a deliberation cost
     that rises as the budget runs out.
6. **Speech**: Broca's area or the prefrontal cortex produces the reply from only what reached
   awareness.
7. **Encoding**: the exchange is stored in episodic memory, weighted by importance, arousal,
   acetylcholine and dopamine.
