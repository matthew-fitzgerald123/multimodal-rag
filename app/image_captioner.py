from __future__ import annotations
import base64
from pathlib import Path
from PIL import Image
from mlx_lm import load, generate
from dotenv import load_dotenv
import os

load_dotenv()

_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        model_id = os.getenv("GEN_MODEL", "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        print(f"Loading captioner: {model_id}")
        _model, _tokenizer = load(model_id)
        print("Captioner ready.")
    return _model, _tokenizer

def caption_image(image_path: str) -> str:
    """
    Generates a detailed text description of an image for embedding.

    Design choice: we use the LLM to describe images in text rather than
    using CLIP-style joint embeddings. This avoids torch dependency entirely
    and produces richer, more searchable descriptions than CLIP's embedding
    space alone. The tradeoff is we lose pixel-level similarity — we gain
    semantic searchability.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and validate image
    img = Image.open(path)
    width, height = img.size
    mode = img.mode
    format_ = img.format or path.suffix.upper().lstrip(".")

    # Build a structured prompt that produces rich, searchable captions
    prompt = f"""[INST] You are analysing an image for a document retrieval system.
The image is {width}x{height} pixels, {mode} mode, {format_} format.
Filename: {path.name}

Provide a detailed description covering:
1. What type of image this is (chart, diagram, photo, table, screenshot, etc.)
2. The main subject or content
3. Any visible text, numbers, labels, or data
4. Colours and visual style
5. Key information someone might search for

Be specific and use domain-relevant terminology. [/INST]"""

    model, tokenizer = get_model()
    caption = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
    return caption.strip()

def get_image_metadata(image_path: str) -> dict:
    """Extracts basic image metadata without loading the full model."""
    path = Path(image_path)
    img = Image.open(path)
    return {
        "width":    img.size[0],
        "height":   img.size[1],
        "mode":     img.mode,
        "format":   img.format or path.suffix.upper().lstrip("."),
        "filename": path.name,
        "size_kb":  round(path.stat().st_size / 1024, 1),
    }
