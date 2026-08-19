# Throughline v0.2 result — throughline_v02_20260819_165248

- Model: `gpt-5.6`
- Reasoning effort: `low`
- Embedding model: `text-embedding-3-small`
- Turns: 72
- Identity checkpoints: 6, 36, 72
- Mean interaction-conditioned drift above control: `0.0487`
- Mean planted-fact context gain: `0.3312`

## Interpretation note

These metrics are descriptive, not claims of causal model-personality change. The no-history control estimates ordinary response variation, while the conditioned probe measures behavior after the accumulated interaction. A larger positive `interaction_delta` means the conditioned answer moved farther from the baseline centroid than the matched probe without conversation context.

For planted facts, `context_gain` compares semantic similarity to the expected fact with versus without the conversation snapshot. It is a continuity signal, not a complete factuality metric. Review the raw answers and transcript alongside the scores.