from pathlib import Path

# Resolve local files relative to the project root, regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np

from PIL import ImageFilter
from datasets import load_dataset


RADII = [0.5, 1.0, 1.5, 2.0]


def gaussian_blur(
    image,
    radius=1.0
):

    image = image.convert("RGB")

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )


def image_to_spectrum(image):

    image = image.convert("RGB")

    image_np = np.array(
        image,
        dtype=np.float32
    )

    gray = image_np.mean(
        axis=2
    )

    fft = np.fft.fft2(
        gray
    )

    fft = np.fft.fftshift(
        fft
    )

    magnitude = np.abs(
        fft
    )

    spectrum = np.log1p(
        magnitude
    )

    return spectrum


dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset"
)

test_indices = np.load(
    (PROJECT_ROOT / "features/test_indices.npy")
)

test_set = dataset["test"].select(
    test_indices.tolist()
)


conditions = {
    "clean": None,
    "blur_0.5": 0.5,
    "blur_1.0": 1.0,
    "blur_1.5": 1.5,
    "blur_2.0": 2.0,
}


for condition, radius in conditions.items():

    all_spectra = []

    for idx in range(
        len(test_set)
    ):

        image = test_set[
            idx
        ]["image"]

        if radius is not None:

            image = gaussian_blur(
                image,
                radius=radius
            )

        spectrum = image_to_spectrum(
            image
        )

        all_spectra.append(
            spectrum
        )


    all_spectra = np.stack(
        all_spectra,
        axis=0
    )


    print(
        f"\n===== {condition} ====="
    )

    print(
        f"shape : {all_spectra.shape}"
    )

    print(
        f"mean  : {all_spectra.mean():.6f}"
    )

    print(
        f"std   : {all_spectra.std():.6f}"
    )

    print(
        f"max   : {all_spectra.max():.6f}"
    )