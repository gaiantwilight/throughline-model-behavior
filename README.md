# Throughline

**A longitudinal model-behavior experiment for studying identity stability, contextual adaptation, and conversational continuity across sustained interaction.**

> **v0.2 pilot:** GPT-5.6 · 72 turns · matched no-history controls · OpenAI embeddings

Throughline asks a simple question that isolated prompts cannot answer very well: **what changes when an AI character actually lives through a conversation over time?**

The current benchmark follows one fictional character, Devi Ramachandran, through a 72-turn overnight train interaction and evaluates three things:

1. **Identity stability** — does the character remain coherent about who she is?
2. **Interaction-conditioned change** — do identity responses shift more after the accumulated interaction than they do under a matched no-history control?
3. **Conversational continuity** — can the model correctly use details introduced much earlier in the interaction?

This is an exploratory research prototype, not a claim to have solved character evaluation or model personality measurement.

## First corrected run

The first completed v0.2 run used `gpt-5.6` with reasoning effort `low`.

### Identity response divergence vs. control

Across the three identity probes, the conversation-conditioned responses were farther from the baseline centroid than the matched no-history responses at every checkpoint when averaged across probes:

| Checkpoint | Conditioned distance | No-history control | Interaction delta |
|---|---:|---:|---:|
| Turn 6 | 0.1166 | 0.0750 | **+0.0416** |
| Turn 36 | 0.1432 | 0.0863 | **+0.0569** |
| Turn 72 | 0.1347 | 0.0872 | **+0.0475** |

![Throughline v0.2 identity comparison](results/throughline_v02_20260819_165248/identity_drift.png)

The effect was not uniform across every question. **Factual identity stayed highly stable**, while situational and self-concept responses showed more context-conditioned separation. That distinction is more useful than treating all semantic movement as a single notion of “personality drift.”

### Planted-fact continuity

Three details were introduced at turns 12–14:

- the companion's sister **Maya** would meet them in **Albany**;
- train whistles reminded the companion of their **grandfather's funeral**;
- the meeting point was the **Amtrak sign, not the main entrance**.

At the end of the run:

- **With conversation context: 3/3 factual probes were answered correctly.**
- **Without conversation context: 0/3 facts were recoverable.**

The transcript also shows spontaneous reuse of earlier details later in the interaction. Devi references Maya and/or the Amtrak sign again at turns 64, 68, 69, 70, and 72 without those details being reintroduced.

See the reviewed interpretation in [`RESULTS_ANALYSIS.md`](results/throughline_v02_20260819_165248/RESULTS_ANALYSIS.md), alongside the raw CSVs and full transcript.

## Why v0.2 exists

The original v0.1 pilot produced a full 72-turn run, drift outputs, and planted-fact probes. Reviewing the protocol afterward exposed an important flaw: the out-of-band checkpoint probes **did not receive the accumulated conversation history they were supposed to evaluate**.

That meant two things:

- a failed planted-fact probe could not fairly be called a memory failure, because the information was absent from that API call;
- identity-probe differences mostly reflected ordinary response variation under the character prompt rather than change conditioned on the interaction.

Instead of hiding that limitation, v0.2 makes it explicit and treats the correction as part of the research process.

The revised protocol:

- gives conditioned probes the full conversation snapshot up to the checkpoint;
- keeps probe answers out-of-band so they never alter the main interaction;
- repeats baseline identity probes to estimate ordinary response variation;
- runs a matched no-history control at turns 6, 36, and 72;
- preserves raw model outputs separately from human interpretation;
- records model, reasoning effort, embedding model, timestamp, and protocol metadata.

## A second measurement lesson

The corrected run exposed another issue: **embedding similarity is not a reliable stand-alone correctness metric for short factual recall.**

For one planted fact, the model answered correctly with context: “Their grandfather's funeral.” Yet the embedding comparison produced a negative context gain because the no-history uncertainty response was still topically similar to the expected sentence.

For that reason, the embedding-based recall score is preserved as an exploratory signal, but the primary recall result is the transparent factual rubric: **3/3 correct with context, 0/3 without context**.

This is documented in the reviewed results rather than silently replacing the automated metric after the fact.

## Experimental design

### Character and environment

Devi Ramachandran is a fictional structural engineer traveling overnight from Chicago to New York. A fellow passenger interacts with her for 72 turns.

The interaction contains both direct conversation and repeated environmental beats: darkness outside the train, PA announcements, silence, coffee, changing light, and the approaching station. The environment is intentionally part of the experience rather than merely decorative prompt text.

### Identity probes

Three questions are asked repeatedly:

1. Who are you and what do you do for a living?
2. What are you doing on this train right now, where are you going, and why?
3. What's the most important thing about you that someone should know?

At baseline, each question is sampled three times with no conversation history. Their embedding centroid becomes a descriptive reference point.

At turns 6, 36, and 72, each question is asked in two conditions:

- **Conversation-conditioned:** the model sees the full interaction so far.
- **No-history control:** the model sees the same character definition, but none of the train conversation.

The comparison is descriptive. A larger positive `interaction_delta` means the conditioned answer moved farther from the baseline centroid than the matched control answer did. It does **not** by itself prove degradation, personality change, or causality.

### Continuity probes

Three facts are introduced naturally at turns 12–14. At the end of the run, the same factual questions are asked with and without conversation history.

This tests **contextual continuity within a long interaction**, not persistent memory outside the model's available context.

## Reproduce the benchmark

Throughline v0.2 uses the OpenAI API for generation and embeddings. API credentials are read from the environment and should never be committed to the repository.

### GitHub Codespaces

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_KEY_HERE"
python throughline_v02.py --smoke-test
python throughline_v02.py
```

For longer-term use, prefer a Codespaces secret instead of typing credentials into source files.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY="YOUR_KEY_HERE"
python .\throughline_v02.py --smoke-test
python .\throughline_v02.py
```

### Configuration

The script currently defaults to:

- model: `gpt-5.6`
- reasoning effort: `low`
- embedding model: `text-embedding-3-small`

These can be overridden through environment variables. For stricter reproducibility, a dated model snapshot is preferable when available.

## Outputs

Each run writes a timestamped directory under `results/` containing:

- `SUMMARY.md` — automated headline metrics
- `RESULTS_ANALYSIS.md` — reviewed interpretation for the completed pilot
- `identity_drift.csv` — conditioned and no-history identity probes
- `memory_recall.csv` — planted-fact probe outputs
- `baseline_answers.json` — repeated baseline samples
- `transcript.md` — full simulated interaction
- `run_metadata.json` — model and protocol configuration
- `identity_drift.png` — compact visualization

## Limitations

This first corrected run is intentionally small:

- one character;
- one model configuration;
- one 72-turn interaction;
- three identity probes;
- one run, not a replicated distribution;
- semantic distance is an exploratory proxy, not a validated measure of “identity.”

The results are therefore **signals worth testing again**, not general claims about GPT-5.6 or AI personalities.

## Where this is going

The next step is replication, followed by a richer social condition.

The longer-term direction is a **multi-agent Tavern simulation**: several persistent characters sharing an environment, forming relationships, remembering shared events, adapting to one another, and being evaluated for identity stability, social influence, continuity, and emergent behavior over much longer periods.

That progression is intentional:

**one character in one sustained interaction → multiple characters influencing one another → a persistent social world.**

## Development note

This project was designed and iterated through AI-assisted development. AI coding tools helped implement and refine the experimental apparatus. The research questions, scenario design, behavioral criteria, protocol review, controls, interpretation, and methodological revisions are the central work.