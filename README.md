# 🎬 Hybrid Movie Recommender System

A high-performance recommendation engine combining **Content-Based Filtering** and **Collaborative Filtering**. This project features a robust **FastAPI** backend and an interactive **Streamlit** frontend, fully containerized with **Docker**.

---

## 📊 Dataset & Training

The model was trained using [The Movies Dataset](https://www.kaggle.com/code/rounakbanik/movie-recommender-systems/input) from Kaggle. 

**Implementation Note:** While the raw data was sourced from Kaggle, the recommendation engine was developed entirely from scratch. The dataset is inherently sparse, so a custom hybrid approach was engineered to overcome this. The entire deployment architecture (FastAPI + Streamlit + Docker) is a custom build and does not rely on the original Kaggle notebooks.

Due to file size constraints, the raw data files (`movies_metadata.csv`, `ratings.csv`, `credits.csv`, etc.) and the initial EDA notebooks are not included in this deployment repository.

---

## 🛠️ Local Setup & Installation

### 1. Environment Setup
It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
The `start.sh` script launches both the API and the Dashboard simultaneously.

```bash
# Grant execution permissions
chmod +x start.sh

# Run the system
./start.sh
```

---

## 🐳 Docker Deployment

Build and run the entire stack using Docker to ensure environment consistency.

### 1. Build the Image
```bash
docker build -t hybrid-recommender .
```

### 2. Run the Container
Map both ports (8000 for FastAPI, 8501 for Streamlit):
```bash
docker run -p 8000:8000 -p 8501:8501 hybrid-recommender
```

---

## 📂 Project Architecture

```text
.
├── backend/
│   ├── main.py                     # FastAPI server and API endpoints
│   ├── exp.py                      # Backend utilities/experiments
│   └── model/
│       └── hybrid_recommender.pkl  # Core ML model 
├── frontend/
│   ├── app.py                      # Streamlit interactive dashboard
│   └── config/
│       └── title.pkl               # Cached titles/configurations
├── Dockerfile                      # Container definition for both services
├── requirements.txt                # Project dependencies
└── start.sh                        # Shell script to boot services concurrently
```

## 📡 Access
Once the services are running, you can access them at:
* **Streamlit UI:** [http://localhost:8501](http://localhost:8501)
* **API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ⚠️ Important Notes

* **Large Files:** The core ML model (`hybrid_recommender.pkl`) is managed via **Git LFS**. Ensure Git LFS is installed before pulling.
* **Storage:** Large backup files (e.g., `hybrid_recommender_backup.pkl`) are explicitly ignored via `.gitignore` to prevent disk space issues and repository bloat.
