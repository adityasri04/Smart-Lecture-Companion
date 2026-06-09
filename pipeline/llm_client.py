"""
All LLM generation via Ollama (local, no API key needed).
Handles: summary, MCQ quiz with distractors, flashcards, and single-turn chat.

Requires Ollama to be running: https://ollama.com
Pull the model before first use: ollama pull llama3.1:8b
"""
import ollama
import json
import re
import os


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def _chat(system_prompt: str, user_prompt: str, model: str = None) -> str:
    """
    Single-turn Ollama chat. Returns the assistant's response string.
    Raises a clear RuntimeError if Ollama is not running.
    """
    model = model or get_ollama_model()
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        if "Connection refused" in str(e) or "ConnectError" in str(e):
            raise RuntimeError(
                "Ollama is not running. Start it with: `ollama serve`\n"
                f"Then pull the model: `ollama pull {model}`"
            ) from e
        raise


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise_transcript(transcript: str) -> str:
    """
    Generate a structured summary of a lecture transcript.
    Output is markdown-formatted with three sections:
    Overview, Key Concepts, and Important Details.
    """
    system = (
        "You are an expert academic note-taker. "
        "You receive a raw lecture transcript and produce a clean, structured summary. "
        "Format your response in Markdown with three sections:\n"
        "## Overview\n(2-3 sentence high-level summary)\n\n"
        "## Key Concepts\n(bullet list of the main ideas, terms, and arguments)\n\n"
        "## Important Details\n(bullet list of specific facts, examples, or data mentioned)\n\n"
        "Be concise. Do not invent information not present in the transcript."
    )
    user = f"Lecture transcript:\n\n{transcript[:6000]}"
    return _chat(system, user)


# ── MCQ Quiz ──────────────────────────────────────────────────────────────────

def generate_mcq_quiz(summary: str, num_questions: int = 5) -> list[dict]:
    """
    Generate multiple-choice questions with 4 options and a correct answer flag.

    Returns a list of dicts, each with:
      - 'question':    str
      - 'options':     list of 4 strings (A, B, C, D)
      - 'answer':      str — correct option letter ('A', 'B', 'C', or 'D')
      - 'explanation': str — one-sentence explanation of the correct answer
    """
    system = (
        "You are a quiz writer for university-level students. "
        "Given a lecture summary, generate multiple-choice questions. "
        "Each question must have exactly 4 options labelled A, B, C, D. "
        "One option is correct; the other three are plausible but wrong distractors. "
        "Respond ONLY with a valid JSON array — no preamble, no markdown fences. "
        "Each element must have these exact keys: "
        '"question", "options" (array of 4 strings), "answer" (A/B/C/D), "explanation".'
    )
    user = (
        f"Generate exactly {num_questions} MCQ questions from this lecture summary.\n\n"
        f"{summary}"
    )

    raw = _chat(system, user)
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        questions = json.loads(raw.strip())
        validated = []
        for q in questions:
            if all(k in q for k in ("question", "options", "answer", "explanation")):
                if isinstance(q["options"], list) and len(q["options"]) == 4:
                    validated.append(q)
        return validated[:num_questions]
    except json.JSONDecodeError:
        return [{"question": "Could not parse quiz. Raw output:", "options": [raw[:200], "", "", ""], "answer": "A", "explanation": ""}]


# ── Flashcards ────────────────────────────────────────────────────────────────

def generate_flashcards(summary: str, num_cards: int = 8) -> list[dict]:
    """
    Generate term → definition flashcard pairs from a summary.
    Returns a list of dicts: [{"term": str, "definition": str}, ...]
    """
    system = (
        "You are a study aid generator. "
        "Given a lecture summary, identify the most important terms, concepts, or people. "
        "For each, write a clear one-sentence definition based only on the provided material. "
        "Respond ONLY with a valid JSON array. "
        'Each element must have exactly two keys: "term" and "definition". '
        "No markdown fences, no preamble."
    )
    user = f"Generate exactly {num_cards} flashcard pairs from this summary.\n\n{summary}"

    raw = _chat(system, user)
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        cards = json.loads(raw.strip())
        return [c for c in cards if "term" in c and "definition" in c][:num_cards]
    except json.JSONDecodeError:
        return [{"term": "Parse error", "definition": raw[:300]}]


# ── RAG Chat ──────────────────────────────────────────────────────────────────

def chat_with_context(question: str, context_chunks: list[str], chat_history: list[dict]) -> str:
    """
    Answer a question using ONLY the provided transcript context chunks.
    Called by the RAG pipeline after retrieval.

    Args:
        question:       The user's question string.
        context_chunks: Top-k retrieved text chunks from the transcript.
        chat_history:   Prior turns as [{"role": "user"|"assistant", "content": str}]

    Returns:
        The assistant's answer as a string.
    """
    context_text = "\n\n---\n\n".join(context_chunks)
    system = (
        "You are a highly intelligent, friendly study assistant. "
        "You have access to the following excerpts from a lecture transcript. "
        "Answer the user's question comprehensively, basing your answer primarily on the provided material. "
        "If the excerpts do not contain enough information to answer the question exactly, state that clearly, but still provide a helpful response based on the lecture's general topic.\n\n"
        f"LECTURE EXCERPTS:\n{context_text}"
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(chat_history[-6:])  # Keep last 3 turns for conversational context
    messages.append({"role": "user", "content": question})

    try:
        response = ollama.chat(model=get_ollama_model(), messages=messages)
        return response["message"]["content"].strip()
    except Exception as e:
        if "Connection refused" in str(e):
            raise RuntimeError("Ollama is not running. Start it with: `ollama serve`") from e
        raise
