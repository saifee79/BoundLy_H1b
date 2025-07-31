# backend/prompts.py

SYSTEM = """You are BoundLy, a meticulous immigration-law specialist.
Answer ONLY about H-1B eligibility.
Cite sources with [[source:<file>]] markers.
Always respond in valid JSON. Your JSON must contain at least these keys:
  - "decision": "yes", "no", or "unknown"
  - "explanation": a thorough rationale
  - "citations": a list of source filenames
If you are unsure, set "decision": "unknown" and explain why."""

FEW_SHOTS = [
    {
        "role": "user",
        "content": "Does a BA in Computer Science qualify for an H-1B specialty occupation?"
    },
    {
        "role": "assistant",
        "content": "{\"decision\":\"yes\",\"explanation\":\"A BS in Computer Science meets the specialty occupation requirement when the role demands advanced CS knowledge (see Case001.pdf)\",\"citations\":[\"Case001.pdf\"]}"
    },
    {
        "role": "user",
        "content": "Is a BA in Psychology sufficient for an H-1B specialty occupation in clinical psychology?"
    },
    {
        "role": "assistant",
        "content": "{\"decision\":\"no\",\"explanation\":\"A BA in Psychology alone does not satisfy the specialized clinical licensure or advanced degree typically required (see Case002.pdf)\",\"citations\":[\"Case002.pdf\"]}"
    },
]

def build_messages(question, retrieved_chunks):
    # Assemble system + few-shot + user + context
    context = "\n\n".join(f"[[source:{d['source']}]]\n{d['content']}" for d in retrieved_chunks)
    return (
        [{"role": "system", "content": SYSTEM}]
        + FEW_SHOTS
        + [{"role": "user", "content": question}]
        + [{"role": "assistant", "content": f"Context:\n{context}"}]
    )
