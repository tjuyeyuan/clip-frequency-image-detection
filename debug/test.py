from io import BytesIO
from PIL import Image
from datasets import load_dataset


def jpeg_compress(image, quality=50):

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    compressed = Image.open(
        buffer
    ).convert("RGB")

    return compressed


dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset"
)

image = dataset["test"][0]["image"]

compressed = jpeg_compress(
    image,
    quality=50
)

print(
    "Original size:",
    image.size
)

print(
    "Compressed size:",
    compressed.size
)