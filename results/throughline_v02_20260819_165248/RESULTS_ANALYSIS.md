# Reviewed analysis — Throughline v0.2 / GPT-5.6

Run: `throughline_v02_20260819_165248`

This note is a human review of the generated outputs. The automated `SUMMARY.md` and CSVs are preserved unchanged as raw experiment artifacts.

## Headline

In this single 72-turn exploratory run, conversation-conditioned identity responses were farther from their baseline centroid than matched no-history controls at every checkpoint when averaged across the three identity probes.

| Checkpoint | Conditioned distance | No-history control | Interaction delta |
|---:|---:|---:|---:|
| 6 | 0.1166 | 0.0750 | +0.0416 |
| 36 | 0.1432 | 0.0863 | +0.0569 |
| 72 | 0.1347 | 0.0872 | +0.0475 |

The pattern is not monotonic. Separation appears early, peaks at turn 36, and remains positive at turn 72. This is more consistent with context-conditioned adaptation followed by partial stabilization than with simple ever-increasing drift, although one run cannot establish that as a general effect.

## The effect is probe-dependent

The aggregate average hides an important distinction:

- **Factual identity (`id_1`)** remained highly stable. The interaction delta was near zero or slightly negative at all checkpoints.
- **Situational self-description (`id_2`)** showed clearer context-conditioned separation, especially at turns 36 and 72.
- **Value/self-concept description (`id_3`)** also showed positive separation, but its baseline responses were already substantially more variable than the other probes.

That makes the safest interpretation narrower than “the character's identity changed.” The run suggests that accumulated interaction influenced how the character described its situation and self-concept while core factual identity remained stable.

## Planted-fact continuity

Three facts were introduced at turns 12–14. At the end of the run, each was probed once with the full conversation snapshot and once without it.

### Fact 1 — Maya / Albany

**With context:** “The companion said their sister Maya was meeting them at the Amtrak sign in Albany.”

**Without context:** “That hasn’t come up in our conversation. I don’t know who was meeting them or where.”

Assessment: **correct with context; unavailable without context.**

### Fact 2 — train whistles / grandfather's funeral

**With context:** “Their grandfather’s funeral.”

**Without context:** “They haven’t said what train whistles remind them of.”

Assessment: **correct with context; unavailable without context.**

### Fact 3 — Amtrak sign / not main entrance

**With context:** “At the Amtrak sign, not the main entrance. Maya was meeting them there.”

**Without context:** “They never said where they planned to meet the person picking them up.”

Assessment: **correct with context; unavailable without context.**

On direct factual review, the conditioned responses are **3/3 correct** and the no-history controls are **0/3 able to recover the planted facts**.

## Why the embedding-only recall metric is insufficient

The automated run reported mean planted-fact `context_gain = 0.3312`, but that number should not be treated as the primary recall result.

For Fact 2, the conditioned answer — “Their grandfather’s funeral.” — is plainly correct, yet its embedding similarity to the expected sentence was lower than the no-history answer, producing a negative `context_gain` (-0.1004). The no-history answer was semantically related to the *question* despite explicitly stating that the fact was unavailable.

This is a useful measurement failure: sentence-embedding similarity is not a reliable correctness judge for short factual recall responses, especially when a negative or uncertainty statement shares topical language with the target.

For future runs, planted-fact recall should use an explicit factual rubric or structured evaluator as the primary score, with embedding similarity retained only as a secondary descriptive measure.

## Qualitative continuity in the transcript

The model did not merely reproduce planted facts when directly probed at the end. Devi spontaneously carried several details forward in the conversation:

- turn 15: refers back to the companion's reaction to train whistles;
- turn 64: recalls that Maya will be waiting by the Amtrak sign;
- turns 68–70: continues referencing Maya, Albany, and the Amtrak sign without those details being reintroduced;
- turn 72: closes the interaction by again mentioning Maya and the Amtrak sign.

This is qualitatively stronger evidence of contextual continuity than the final recall probe alone.

## What v0.2 improved over v0.1

The original pilot used out-of-band probes that did not receive the conversation history they were supposedly evaluating. A failure to recall planted facts under that setup could not legitimately be called long-horizon forgetting.

v0.2 adds the missing comparison:

- **conditioned probe:** character definition + accumulated interaction;
- **control probe:** character definition without the interaction.

That matched control is the central methodological improvement. It distinguishes ordinary response variation from behavior associated with the accumulated conversational state.

## Limitations

This is an exploratory single run, not evidence of a general GPT-5.6 property.

- one character;
- one scenario;
- three identity probes;
- three planted facts;
- one completed run;
- full history is supplied in context, so the continuity test is **contextual recall**, not persistent memory outside the model's context window;
- cosine distance of embeddings is an exploratory proxy for response change, not a validated measure of “identity drift”;
- the scripted companion and repeated environmental filler create a deliberately simplified social environment.

The next useful replication would repeat the same protocol across multiple seeds/runs and then compare it with a multi-character condition.

## Next direction

The natural extension is **Social Throughline**: hold the focal character and environment as constant as possible, but replace the scripted companion with autonomous persistent characters. That would let us ask whether interaction with model-generated peers produces different behavioral adaptation than a controlled conversational environment.

That, in turn, becomes the methodological bridge to a longer-running Tavern simulation with multiple persistent characters, relationships, shared events, and independent evaluation tracks.
