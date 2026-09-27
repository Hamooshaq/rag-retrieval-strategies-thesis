from __future__ import annotations


class PromptBuilder:
    REFUSAL_TOKEN = "INSUFFICIENT_EVIDENCE"

    def build_rag_prompt(self, question: str, context: str) -> str:
        return (
            "You answer a question using only the EVIDENCE block. "
            "Do not use outside knowledge. If the evidence is empty or does not support the answer, "
            f"output exactly {self.REFUSAL_TOKEN}. "
            "Otherwise answer concisely and cite every supporting locator using [page:n] or [slide:n].\n\n"
            f"EVIDENCE:\n{context or '(empty)'}\n\n"
            f"QUESTION:\n{question}\n\nANSWER:"
        )

    def build_no_rag_prompt(self, question: str) -> str:
        return (
            "Answer the question from your internal knowledge. If you are not confident that the answer "
            f"is correct, output exactly {self.REFUSAL_TOKEN}. Do not invent source citations.\n\n"
            f"QUESTION:\n{question}\n\nANSWER:"
        )
