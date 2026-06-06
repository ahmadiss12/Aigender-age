# Age and Gender Prediction AI Demo

A compact Streamlit application that estimates gender and age from an uploaded face image using two pretrained Keras model artifacts.

## What This Project Shows

- Streamlit interface for uploading an image and displaying predictions.
- Separate Keras models for gender classification and age regression.
- Image preprocessing with EXIF correction, RGB conversion, resize, and normalization.
- Confidence display for the gender model output.
- Clear model-card notes instead of unsupported accuracy claims.
- Basic tests for preprocessing and output normalization.

## Tech Stack

- Python 3.11
- Streamlit
- TensorFlow / Keras
- NumPy
- Pillow
- Pytest

## Project Structure

```text
.
+-- app.py              # Streamlit UI
+-- prediction.py       # Preprocessing, model loading, and prediction helpers
+-- age_model.h5        # Age regression model artifact
+-- gender_model.h5     # Gender classification model artifact
+-- tests/              # Lightweight unit tests
+-- requirements.txt
`-- requirements-dev.txt
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
streamlit run app.py
```

## Test

```bash
pytest
python -m compileall app.py prediction.py
```

## Model Card

| Item | Details |
| --- | --- |
| Task | Age estimation and gender classification from a face image |
| Input | RGB image resized to 64 x 64 pixels |
| Output | Gender label, gender confidence, and estimated age |
| Model files | `gender_model.h5`, `age_model.h5` |
| Runtime | TensorFlow / Keras |
| Metrics | Not available in the current repository |

## Metrics Status

This repository contains model artifacts but does not include the original training data, evaluation split, training notebook, or validation report. Because of that, the README does not claim accuracy, mean absolute error, or fairness metrics.

For a stronger job-ready AI portfolio project, add:

- Dataset description and preprocessing pipeline.
- Training notebook or training script.
- Gender accuracy, precision, recall, F1 score, and confusion matrix.
- Age mean absolute error and age-bucket error analysis.
- Evaluation examples across lighting, pose, image quality, and demographic groups.
- A short limitations section explaining bias and responsible-use concerns.

## Limitations

This demo should not be used for identity verification, hiring decisions, access control, medical use, or any high-stakes age or gender decision. It is a portfolio demo for model deployment and inference UI work.
