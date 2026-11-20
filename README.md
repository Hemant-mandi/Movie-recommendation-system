# 🎬 OTT Platform — Personalized Recommendation System

**Subject:** Data Analytics (BCS714D)

## Overview
A big data analytics project that analyzes user viewing history, ratings, and preferences on an OTT platform to build a personalized movie and web series recommendation system.

## Tech Stack
- **Big Data:** Apache Hadoop (HDFS + MapReduce)
- **Programming:** Java (MapReduce), Python (Analytics & ML)
- **ML Libraries:** Scikit-learn, Surprise (SVD, KNN)
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Dataset:** MovieLens 25M + TMDB 5000

## Project Structure
```
Big Data Project/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned/transformed data
├── hadoop/
│   ├── src/                    # Java MapReduce source code
│   ├── input/                  # MapReduce input files
│   └── output/                 # MapReduce output files
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── content_based.py        # Content-based filtering
│   ├── collaborative.py        # Collaborative filtering (SVD)
│   ├── hybrid.py               # Hybrid recommender
│   └── utils.py                # Helper functions
├── app/
│   └── streamlit_app.py        # Streamlit dashboard
├── reports/
│   └── project_report.pdf      # Final report
├── requirements.txt
└── README.md
```

## Setup
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## How to Run
```bash
# Start Streamlit dashboard
streamlit run app/streamlit_app.py
```
