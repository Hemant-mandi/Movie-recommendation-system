# Movie Recommendation System

A movie recommender built on the MovieLens dataset. It combines a popularity baseline, content-based filtering (TF-IDF over genres and tags), and collaborative filtering with SVD matrix factorisation. A hybrid engine blends the last two. A Hadoop MapReduce job computes genre statistics, and a Streamlit dashboard presents the results.

## Results

Collaborative filtering (SVD) was evaluated with 5-fold cross-validation on MovieLens ml-latest-small:

| Model | Evaluation | Score |
|---|---|---|
| SVD collaborative filtering (30 latent factors, 15 epochs) | 5-fold CV RMSE | **0.8749** |

See [`reports/project_report.md`](reports/project_report.md) for the full write-up, including the MAE and the comparison with KNN.

## Features

- **Popularity baseline** for new users and cold-start cases
- **Content-based recommendations** using TF-IDF and cosine similarity over genres and user tags
- **Collaborative filtering** with SVD (Surprise library)
- **Hybrid recommender** that blends collaborative and content scores (`alpha` controls the weight)
- **Hadoop MapReduce jobs** that compute genre popularity and movie rating statistics
- **Streamlit dashboard** with recommendation tabs and an interactive view of the results

## Tech stack

- **Python:** pandas, NumPy, scikit-learn, scikit-surprise, Matplotlib, Seaborn, Plotly, Streamlit
- **Big data:** Apache Hadoop (MapReduce), Java
- **Dataset:** [MovieLens](https://grouplens.org/datasets/movielens/) (ml-latest-small)

## Project structure

```
.
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
├── src/
│   ├── utils.py                  # Data loading and helpers
│   ├── content_based.py          # TF-IDF content-based recommender
│   ├── collaborative.py          # SVD collaborative filtering
│   └── hybrid.py                 # Hybrid recommender
├── hadoop/
│   ├── src/                      # MapReduce jobs (Java)
│   ├── input/                    # Sample input for the jobs
│   └── pom.xml
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis
├── reports/
│   ├── project_report.md         # Full project report
│   └── evaluator_demo_guide.md   # Walkthrough of the dashboard
├── download_data.py              # Downloads and extracts MovieLens
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/Hemant-mandi/Movie-recommendation-system.git
cd Movie-recommendation-system

# 2. Create and activate a virtual environment
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset into data/raw/
python download_data.py
```

`scikit-surprise` compiles C extensions. On Windows, install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first if the install fails.

## Usage

Launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

Run the recommenders directly:

```bash
python src/collaborative.py
python src/content_based.py
```

Explore the data in Jupyter:

```bash
jupyter notebook notebooks/eda.ipynb
```

Run the Hadoop jobs (requires a local Hadoop installation):

```bash
cd hadoop
mvn package
hadoop jar target/<jar-name>.jar GenrePopularity input/movies.csv output/genres
```

## Dataset

The dataset is not stored in this repository. `download_data.py` fetches MovieLens ml-latest-small from GroupLens and extracts it into `data/raw/`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).