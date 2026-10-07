from pathlib import Path

# Resolve local files relative to the project root, regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

texts = [
    "a real photograph",
    "an AI-generated image"
]

image_paths = [
    (PROJECT_ROOT / "data/real.jpg"),
    (PROJECT_ROOT / "data/ai.jpg")
]

for image_path in image_paths:

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

    print("\nImage:", image_path)

    for text, prob in zip(texts, probs[0]):
        print(
            f"{text}: {prob.item():.4f}"
        )