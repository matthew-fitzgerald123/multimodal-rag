from __future__ import annotations
from mlx_lm import load, generate
from dotenv import load_dotenv
import os

load_dotenv()

class Generator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._model_id = os.getenv(
            "GEN_MODEL",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        )

    def load_model(self):
        print(f"Loading generator: {self._model_id}")
        self.model, self.tokenizer = load(self._model_id)
        print("Generator ready.")

    def answer(self, query: str, chunks: list[dict], max_tokens: int = 512) -> str:
        if self.model is None:
            raise RuntimeError("Generator not loaded")

        # Separate text and image chunks for the prompt
        text_chunks = [c for c in chunks if c.get("modality") != "image"]
        image_chunks = [c for c in chunks if c.get("modality") == "image"]

        context_parts = []

        if text_chunks:
            context_parts.append("Text sources:")
            for i, c in enumerate(text_chunks):
                context_parts.append(f"[T{i+1}] {c['text']}")

        if image_chunks:
            context_parts.append("\nImage sources (described):")
            for i, c in enumerate(image_chunks):
                fname = c.get("metadata", {}).get("filename", f"image_{i+1}")
                context_parts.append(f"[I{i+1}] ({fname}) {c['text']}")

        context = "\n".join(context_parts)

        prompt = f"""[INST] Answer the question using only the provided context.
The context includes both text sources and image descriptions.
If the answer comes from an image, mention which image it came from.
If you cannot answer from the context, say so clearly.

Context:
{context}

Question: {query} [/INST]"""

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return response.strip()

generator = Generator()
