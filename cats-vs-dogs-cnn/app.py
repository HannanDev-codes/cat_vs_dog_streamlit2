"""Streamlit web app: upload a picture, get 'cat' or 'dog' plus a Grad-CAM heatmap."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

# Make `import catsdogs` work no matter where Streamlit is launched from
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from catsdogs.inference import Predictor  # noqa: E402

st.set_page_config(page_title="Cat vs Dog CNN", page_icon="🐱", layout="centered")

# The repo keeps the model at <repo root>/models, while this app lives one level down.
MODEL_CANDIDATES = [
    APP_DIR / "models" / "best_model.pt",
    APP_DIR.parent / "models" / "best_model.pt",
    Path.cwd() / "models" / "best_model.pt",
]


def find_model() -> Path:
    for p in MODEL_CANDIDATES:
        if p.is_file():
            return p
    st.error("Could not find `models/best_model.pt` in the repository.")
    st.stop()


def is_lfs_pointer(path: Path) -> bool:
    """A Git LFS pointer is a ~130-byte text stub, not the real weights."""
    if path.stat().st_size > 1024:
        return False
    return path.read_bytes()[:40].startswith(b"version https://git-lfs")


@st.cache_resource(show_spinner=False)
def load_predictor() -> Predictor:
    path = find_model()
    if is_lfs_pointer(path):
        st.error(
            "`models/best_model.pt` is only a Git LFS pointer, so the real weights were not "
            "downloaded. See the README section on large model files."
        )
        st.stop()
    return Predictor(str(path), device="cpu")


def energy_score(predictor: Predictor, image: Image.Image) -> float:
    """Free-energy score. Lower = looks like training data; higher = likely not a cat/dog."""
    img = ImageOps.exif_transpose(image).convert("RGB")
    x = predictor.tf(img).unsqueeze(0).to(predictor.device)
    with torch.no_grad():
        logits = predictor.model(x)
    return float(-torch.logsumexp(logits, dim=1).item())


def main():
    st.title("🐱 Cat vs Dog 🐶")
    st.write("Upload a photo of a **cat or dog** and the CNN will classify it.")
    st.write("A heatmap shows **where** the model looked (red/yellow = most important).")

    with st.spinner("Loading model..."):
        predictor = load_predictor()

    acc = f"{predictor.val_acc:.1%}" if predictor.val_acc is not None else "n/a"
    st.caption(f"Model: `{predictor.arch}` · Validation accuracy: {acc}")

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider("Confidence threshold", 0.5, 1.0, 0.7, 0.05)
        use_ood = st.checkbox("Reject non cat/dog images (energy OOD filter)", value=True)
        energy_threshold = st.slider(
            "OOD energy threshold (higher = stricter)", -15.0, 15.0, -3.0, 0.5,
            disabled=not use_ood,
        )

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not read that image: {err}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your picture")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing..."):
        energy = energy_score(predictor, image)
        scores, overlay = predictor.predict(image, heatmap=True)

    with col2:
        if use_ood and energy > energy_threshold:
            st.subheader("Result")
            st.warning(
                "This does not look like a cat or a dog "
                f"(energy {energy:.2f} > threshold {energy_threshold:.1f})."
            )
            return

        best_label, best_prob = max(scores.items(), key=lambda kv: kv[1])
        st.subheader("Prediction")
        if best_prob < threshold:
            st.warning(
                f"Low confidence ({best_prob:.1%}). Might be a **{best_label}**, "
                f"but it is below the {threshold:.0%} threshold."
            )
        for label, prob in sorted(scores.items(), key=lambda kv: -kv[1]):
            st.write(f"**{label}**: {prob:.1%}")
            st.progress(float(prob))

        if overlay is not None:
            st.subheader("Where the model looked")
            st.image(overlay, use_container_width=True)
            st.caption("Bright areas pushed the model towards its answer.")

    st.divider()
    st.caption("This model only knows cats and dogs. Other subjects may give unreliable results.")


if __name__ == "__main__":
    main()
