# SkyCity Restaurant Growth Potential Modeling & Strategic Classification

## Setup (once)
    python -m venv .venv
    .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (macOS/Linux)
    pip install -r requirements.txt

## Run order (all commands from the project root, the folder containing app.py)
    python -m src.pipeline      builds data/restaurants_scored.csv and artifacts/
    python -m src.figures       builds figures/*.png
    python -m src.reports       builds the two Word reports in reports/
    python -m pytest -q         runs 16 tests
    streamlit run app.py        opens the dashboard at http://localhost:8501

## Layout
    data/        input CSV and scored output
    src/         config, features, preprocess, modeling, pipeline, figures, reports
    tests/       pytest suite (pipeline and Streamlit app)
    artifacts/   metrics, PCA loadings, cluster summaries, lever importance
    figures/     charts used in the paper
    reports/     research paper and executive summary (.docx)
    app.py       Streamlit dashboard
