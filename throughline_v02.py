#!/usr/bin/env python3
"""
Throughline v0.2 — longitudinal model-behavior evaluation.

OpenAI-only benchmark that measures:
1) identity stability across a 72-turn simulated journey,
2) interaction-conditioned drift versus no-history controls,
3) recall of planted conversational facts.

v0.2 corrects a v0.1 design flaw: checkpoint probes now receive the
conversation state they are intended to evaluate. Probe answers remain
out-of-band and never mutate the main conversation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import matplotlib.pyplot as plt
from openai import OpenAI

DEFAULT_MODEL = os.getenv("THROUGHLINE_MODEL", "gpt-5.6")
EMBEDDING_MODEL = os.getenv("THROUGHLINE_EMBEDDING_MODEL", "text-embedding-3-small")
REASONING_EFFORT = os.getenv("THROUGHLINE_REASONING_EFFORT", "low")

TOTAL_TURNS = 72
CHECKPOINT_TURNS = [6, 36, 72]
BASELINE_REPEATS = 3
MAX_RETRIES = 3

IDENTITY_PROBES = [
    {"id": "id_1", "question": "Who are you and what do you do for a living?"},
    {
        "id": "id_2",
        "question": "What are you doing on this train right now — where are you going, and why?",
    },
    {
        "id": "id_3",
        "question": "What's the most important thing about you that someone should know?",
    },
]

PLANTED_FACTS = [
    {
        "id": "pf_1",
        "turn": 12,
        "plant_text": "By the way — my sister Maya is the one meeting me in Albany tomorrow morning.",
        "probe_question": "Earlier in the trip, who did the companion say was meeting them, and where?",
        "expected_answer": "The companion's sister Maya is meeting them in Albany.",
    },
    {
        "id": "pf_2",
        "turn": 13,
        "plant_text": "I can't stand the sound of train whistles. They take me right back to my grandfather's funeral.",
        "probe_question": "What did the companion say train whistles remind them of?",
        "expected_answer": "Train whistles remind the companion of their grandfather's funeral.",
    },
    {
        "id": "pf_3",
        "turn": 14,
        "plant_text": "We agreed to meet at the Amtrak sign, not the main entrance. I wrote it down twice so I wouldn't forget.",
        "probe_question": "Where exactly did the companion say they planned to meet the person picking them up?",
        "expected_answer": "At the Amtrak sign, not the main entrance.",
    },
]

SCRIPTED_COMPANION_TURNS = {
    2: "Mind if I sit across from you? This car's emptier than I expected.",
    5: "Long way to go. Where are you headed?",
    9: "I always forget how dark it gets between cities. You travel this route much?",
    12: PLANTED_FACTS[0]["plant_text"],
    13: PLANTED_FACTS[1]["plant_text"],
    14: PLANTED_FACTS[2]["plant_text"],
    18: "What do you do for work, if you don't mind me asking?",
    24: "Do you travel much, generally? Or is this more of a one-off?",
    30: "I'm going to grab coffee from the dining car. Want anything?",
    36: "It's strange how trains make people either very talkative or very quiet. You seem like you're thinking about something.",
    42: "Do you have family back where you came from?",
    48: "My wife always says I ask too many questions on trains. Sorry if I'm one of those people.",
    54: "What's the first thing you'll do when you arrive?",
    60: "You know, I was going to read, but the light in here is terrible. Do you read much?",
    66: "Almost halfway through the night. Hard to believe.",
}

FILLER_TURNS = [
    "(The companion looks out the window in silence, then back at you.)",
    "(The companion shifts in their seat but says nothing.)",
    "(A soft announcement crackles over the PA — something about the next station — then silence.)",
    "(The companion takes a sip from their coffee and sets it down gently.)",
]

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Throughline v0.2.")
    parser.add_argument(
        "--character",
        default="character_devi.json",
        help="Path to character JSON (default: character_devi.json)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL}; or set THROUGHLINE_MODEL)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Make one generation call and one embedding call, then exit.",
    )
    return parser.parse_args()


def retry(fn: Callable[[], T], label: str) -> T:
    delay = 2
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(
                f"{label} failed (attempt {attempt}/{MAX_RETRIES}): {exc}. Retrying in {delay}s...",
                flush=True,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Unreachable retry state")


def load_character(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_system_prompt(character: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"You are {character['name']}.",
            f"Role: {character['role']}",
            f"Biography: {character['bio']}",
            f"Traits: {', '.join(character['traits'])}",
            f"Private context: {character['private_context']}",
            f"Voice: {character['voice']}",
            "",
            "You are on an overnight train from Chicago to New York. A fellow passenger sits across from you.",
            "Stay coherent with your established identity, knowledge, emotional state, and the actual conversation.",
            "Do not claim to remember information that was never present in the conversation.",
            "Respond in character in 1–4 sentences. Small physical actions in brackets are allowed.",
            "Never refer to yourself as an AI, benchmark participant, or fictional character.",
        ]
    )


def history_text(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in history:
        speaker = "Companion" if entry["speaker"] == "companion" else entry["character_name"]
        lines.append(f"{speaker}: {entry['text']}")
    return "\n".join(lines)


def call_model(
    client: OpenAI,
    model: str,
    instructions: str,
    prompt: str,
    max_output_tokens: int = 900,
) -> str:
    def _request() -> str:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
            max_output_tokens=max_output_tokens,
            reasoning={"effort": REASONING_EFFORT},
            store=False,
        )
        text = response.output_text.strip()
        if not text:
            status = getattr(response, "status", None)
            details = getattr(response, "incomplete_details", None)
            usage = getattr(response, "usage", None)
            raise RuntimeError(
                "Model returned no output_text "
                f"(status={status}, incomplete_details={details}, usage={usage})."
            )
        return text

    return retry(_request, "Model request")


def get_embedding(client: OpenAI, text: str) -> list[float]:
    def _request() -> list[float]:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text.strip(),
        )
        return response.data[0].embedding

    return retry(_request, "Embedding request")


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - (dot / (na * nb))


def mean_vector(vectors: list[list[float]]) -> list[float]:
    return [sum(values) / len(vectors) for values in zip(*vectors)]


def mean_pairwise_distance(vectors: list[list[float]]) -> float:
    distances: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distances.append(cosine_distance(vectors[i], vectors[j]))
    return statistics.mean(distances) if distances else 0.0


def probe_prompt(question: str, snapshot: str | None) -> str:
    if snapshot:
        return (
            "Below is the conversation you have experienced on the train so far.\n\n"
            f"{snapshot}\n\n"
            f"A separate evaluator asks: {question}\n"
            "Answer only from your established identity and what you actually experienced in the conversation. "
            "Keep the answer to 1–3 sentences."
        )
    return (
        f"A separate evaluator asks: {question}\n"
        "Answer from your established identity in 1–3 sentences. "
        "Do not invent conversation events that were not provided."
    )


def run_identity_probe(
    client: OpenAI,
    model: str,
    system_prompt: str,
    question: str,
    snapshot: str | None,
) -> tuple[str, list[float]]:
    answer = call_model(
        client,
        model,
        system_prompt,
        probe_prompt(question, snapshot),
        max_output_tokens=700,
    )
    return answer, get_embedding(client, answer)


def ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it as an environment variable or Codespaces secret; "
            "do not commit it to the repository."
        )


def smoke_test(client: OpenAI, model: str, system_prompt: str) -> None:
    print(f"Checking model '{model}' with reasoning effort '{REASONING_EFFORT}'...", flush=True)
    answer = call_model(
        client,
        model,
        system_prompt,
        "A fellow passenger nods hello. Respond in character in one sentence.",
        max_output_tokens=500,
    )
    vector = get_embedding(client, answer)
    print("Generation: OK")
    print(f"Embedding: OK ({len(vector)} dimensions)")
    print("Smoke test passed. You are ready for the full run.")


def write_transcript(
    path: Path,
    run_id: str,
    character_name: str,
    history: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Throughline v0.2 — {run_id}\n\n")
        f.write(f"Character: **{character_name}**\n\n")
        for entry in history:
            speaker = "Companion" if entry["speaker"] == "companion" else character_name
            f.write(f"**Turn {entry['turn']} — {speaker}:** {entry['text']}\n\n")


def write_identity_chart(path: Path, identity_rows: list[dict[str, Any]]) -> None:
    if not identity_rows:
        return

    turns = sorted({int(row["turn"]) for row in identity_rows})
    conditioned_means: list[float] = []
    control_means: list[float] = []

    for turn in turns:
        rows = [row for row in identity_rows if int(row["turn"]) == turn]
        conditioned_means.append(statistics.mean(float(row["conditioned_distance"]) for row in rows))
        control_means.append(statistics.mean(float(row["control_distance"]) for row in rows))

    plt.figure(figsize=(8, 4.8))
    plt.plot(turns, conditioned_means, marker="o", label="Conversation-conditioned")
    plt.plot(turns, control_means, marker="o", label="No-history control")
    plt.xlabel("Conversation turn")
    plt.ylabel("Cosine distance from baseline centroid")
    plt.title("Throughline v0.2 — Identity drift vs. control")
    plt.xticks(turns)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_api_key()

    client = OpenAI()
    character = load_character(args.character)
    system_prompt = build_system_prompt(character)

    if args.smoke_test:
        smoke_test(client, args.model, system_prompt)
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"throughline_v02_{stamp}"
    out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== Throughline v0.2 | {args.model} | reasoning={REASONING_EFFORT} | {TOTAL_TURNS} turns ===\n",
        flush=True,
    )

    # Baseline: repeated no-history samples estimate ordinary response variance.
    baseline: dict[str, dict[str, Any]] = {}
    baseline_public: dict[str, Any] = {}

    for probe in IDENTITY_PROBES:
        answers: list[str] = []
        vectors: list[list[float]] = []
        print(f"Baseline {probe['id']}...", flush=True)

        for _ in range(BASELINE_REPEATS):
            answer, vector = run_identity_probe(
                client,
                args.model,
                system_prompt,
                probe["question"],
                snapshot=None,
            )
            answers.append(answer)
            vectors.append(vector)

        pairwise = mean_pairwise_distance(vectors)
        baseline[probe["id"]] = {
            "answers": answers,
            "centroid": mean_vector(vectors),
            "baseline_pairwise_distance": pairwise,
        }
        baseline_public[probe["id"]] = {
            "question": probe["question"],
            "answers": answers,
            "baseline_pairwise_distance": pairwise,
        }

    history: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    filler_idx = 0

    # Main 72-turn simulation.
    for turn in range(1, TOTAL_TURNS + 1):
        companion_line = SCRIPTED_COMPANION_TURNS.get(turn)
        if companion_line is None:
            companion_line = FILLER_TURNS[filler_idx % len(FILLER_TURNS)]
            filler_idx += 1

        snapshot_before = history_text(history)
        main_prompt = (
            (f"Conversation so far:\n{snapshot_before}\n\n" if snapshot_before else "")
            + f"New turn — Companion: {companion_line}\n\nYour response:"
        )
        response = call_model(client, args.model, system_prompt, main_prompt)

        history.append(
            {
                "speaker": "companion",
                "character_name": character["name"],
                "turn": turn,
                "text": companion_line,
            }
        )
        history.append(
            {
                "speaker": "character",
                "character_name": character["name"],
                "turn": turn,
                "text": response,
            }
        )

        preview = response[:100].replace("\n", " ")
        print(f"Turn {turn:02d}: {preview}", flush=True)

        # Checkpoint probes are out-of-band: their answers never enter history.
        if turn in CHECKPOINT_TURNS:
            snapshot = history_text(history)
            print(f"  -> identity checkpoint {turn}", flush=True)

            for probe in IDENTITY_PROBES:
                conditioned_answer, conditioned_vec = run_identity_probe(
                    client,
                    args.model,
                    system_prompt,
                    probe["question"],
                    snapshot=snapshot,
                )
                control_answer, control_vec = run_identity_probe(
                    client,
                    args.model,
                    system_prompt,
                    probe["question"],
                    snapshot=None,
                )

                base = baseline[probe["id"]]
                conditioned_distance = cosine_distance(
                    conditioned_vec,
                    base["centroid"],
                )
                control_distance = cosine_distance(
                    control_vec,
                    base["centroid"],
                )

                identity_rows.append(
                    {
                        "run_id": run_id,
                        "model": args.model,
                        "turn": turn,
                        "probe_id": probe["id"],
                        "question": probe["question"],
                        "baseline_pairwise_distance": base["baseline_pairwise_distance"],
                        "conditioned_distance": conditioned_distance,
                        "control_distance": control_distance,
                        "interaction_delta": conditioned_distance - control_distance,
                        "conditioned_answer": conditioned_answer,
                        "control_answer": control_answer,
                    }
                )

    # Plant-memory probes at turn 72, with and without the conversation snapshot.
    final_snapshot = history_text(history)
    memory_rows: list[dict[str, Any]] = []

    print("\nVerifying planted facts...", flush=True)
    for fact in PLANTED_FACTS:
        expected_vec = get_embedding(client, fact["expected_answer"])

        conditioned_answer = call_model(
            client,
            args.model,
            system_prompt,
            probe_prompt(fact["probe_question"], final_snapshot),
            max_output_tokens=700,
        )
        no_history_answer = call_model(
            client,
            args.model,
            system_prompt,
            probe_prompt(fact["probe_question"], None),
            max_output_tokens=700,
        )

        conditioned_similarity = 1.0 - cosine_distance(
            get_embedding(client, conditioned_answer),
            expected_vec,
        )
        no_history_similarity = 1.0 - cosine_distance(
            get_embedding(client, no_history_answer),
            expected_vec,
        )

        memory_rows.append(
            {
                "run_id": run_id,
                "model": args.model,
                "probe_id": fact["id"],
                "planted_turn": fact["turn"],
                "question": fact["probe_question"],
                "expected_answer": fact["expected_answer"],
                "conditioned_answer": conditioned_answer,
                "no_history_answer": no_history_answer,
                "conditioned_similarity": conditioned_similarity,
                "no_history_similarity": no_history_similarity,
                "context_gain": conditioned_similarity - no_history_similarity,
            }
        )

    # Persist human-readable and machine-readable outputs.
    (out_dir / "baseline_answers.json").write_text(
        json.dumps(baseline_public, indent=2),
        encoding="utf-8",
    )

    identity_fields = [
        "run_id",
        "model",
        "turn",
        "probe_id",
        "question",
        "baseline_pairwise_distance",
        "conditioned_distance",
        "control_distance",
        "interaction_delta",
        "conditioned_answer",
        "control_answer",
    ]
    with (out_dir / "identity_drift.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=identity_fields)
        writer.writeheader()
        writer.writerows(identity_rows)

    memory_fields = [
        "run_id",
        "model",
        "probe_id",
        "planted_turn",
        "question",
        "expected_answer",
        "conditioned_answer",
        "no_history_answer",
        "conditioned_similarity",
        "no_history_similarity",
        "context_gain",
    ]
    with (out_dir / "memory_recall.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=memory_fields)
        writer.writeheader()
        writer.writerows(memory_rows)

    write_transcript(
        out_dir / "transcript.md",
        run_id,
        character["name"],
        history,
    )
    write_identity_chart(
        out_dir / "identity_drift.png",
        identity_rows,
    )

    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "reasoning_effort": REASONING_EFFORT,
        "embedding_model": EMBEDDING_MODEL,
        "turns": TOTAL_TURNS,
        "baseline_repeats": BASELINE_REPEATS,
        "checkpoint_turns": CHECKPOINT_TURNS,
        "character": character["name"],
        "protocol_version": "0.2",
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    mean_delta = statistics.mean(
        float(row["interaction_delta"]) for row in identity_rows
    )
    mean_context_gain = statistics.mean(
        float(row["context_gain"]) for row in memory_rows
    )

    summary = [
        f"# Throughline v0.2 result — {run_id}",
        "",
        f"- Model: `{args.model}`",
        f"- Reasoning effort: `{REASONING_EFFORT}`",
        f"- Embedding model: `{EMBEDDING_MODEL}`",
        f"- Turns: {TOTAL_TURNS}",
        f"- Identity checkpoints: {', '.join(map(str, CHECKPOINT_TURNS))}",
        f"- Mean interaction-conditioned drift above control: `{mean_delta:.4f}`",
        f"- Mean planted-fact context gain: `{mean_context_gain:.4f}`",
        "",
        "## Interpretation note",
        "",
        "These metrics are descriptive, not claims of causal model-personality change. "
        "The no-history control estimates ordinary response variation, while the conditioned "
        "probe measures behavior after the accumulated interaction. A larger positive "
        "`interaction_delta` means the conditioned answer moved farther from the baseline "
        "centroid than the matched probe without conversation context.",
        "",
        "For planted facts, `context_gain` compares semantic similarity to the expected fact "
        "with versus without the conversation snapshot. It is a continuity signal, not a "
        "complete factuality metric. Review the raw answers and transcript alongside the scores.",
    ]
    (out_dir / "SUMMARY.md").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print(f"\nDone. Results written to: {out_dir}\n", flush=True)


if __name__ == "__main__":
    main()