from __future__ import annotations

import streamlit as st
from PIL import Image, UnidentifiedImageError

from prediction import (
    AGE_MODEL_PATH,
    GENDER_MODEL_PATH,
    IMAGE_SIZE,
    load_prediction_models,
    predict_age_gender,
)


@st.cache_resource(show_spinner="Loading model files...")
def get_models():
    return load_prediction_models()


def render_sidebar() -> None:
    st.sidebar.header("Model card")
    st.sidebar.write("Task: face-based age and gender estimation.")
    st.sidebar.write(f"Input: RGB image resized to {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}.")
    st.sidebar.write("Output: gender class, confidence score, and estimated age.")

    st.sidebar.divider()
    st.sidebar.caption("Model artifacts")
    st.sidebar.code(GENDER_MODEL_PATH.name)
    st.sidebar.code(AGE_MODEL_PATH.name)

    st.sidebar.divider()
    st.sidebar.caption("Evaluation status")
    st.sidebar.write(
        "Training data and validation metrics are not included in this repo yet, "
        "so the demo reports inference confidence without claiming production accuracy."
    )


def render_prediction(image: Image.Image) -> None:
    try:
        gender_model, age_model = get_models()
        result = predict_age_gender(image, gender_model, age_model)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    gender_col, confidence_col, age_col = st.columns(3)
    gender_col.metric("Gender", result.gender)
    confidence_col.metric("Confidence", f"{result.gender_confidence:.1%}")
    age_col.metric("Estimated age", f"{result.age} years")

    st.progress(result.gender_confidence)

    with st.expander("Raw model output"):
        st.write(
            {
                "gender": result.gender,
                "gender_confidence": round(result.gender_confidence, 4),
                "age_display": result.age,
                "age_raw": round(result.raw_age, 4),
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="Age and Gender AI Demo",
        page_icon=None,
        layout="centered",
    )

    st.title("Age and Gender Prediction")
    st.caption("A Streamlit demo powered by two compact Keras models.")
    render_sidebar()

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload a JPG or PNG image to run a prediction.")
        return

    try:
        image = Image.open(uploaded_file)
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image.")
        return

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            render_prediction(image)


if __name__ == "__main__":
    main()
