from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import scipy.sparse
import os
import gc

app = FastAPI(
    title="Hybrid Movie Recommender API", 
    version="2.0.0",
    servers=[{"url": "http://localhost:8000", "description": "Local Development"}]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "hybrid_recommender.pkl")

print("=" * 60)
print("LOADING MODEL...")
print("=" * 60)

bundle = joblib.load(MODEL_PATH)

# Extract only what we need
new_df = bundle["new_df"]
vectors = bundle["vectors"]

# Free memory immediately
del bundle
gc.collect()

print(f"✅ Loaded {len(new_df)} movies")
print(f"✅ Vectors type: {type(vectors).__name__}")

# Verify sparse format
if scipy.sparse.issparse(vectors):
    print(f"✅ Sparse format: {vectors.data.nbytes / 1e6:.2f} MB")
    print(f"✅ Sparsity: {(1 - vectors.nnz / np.prod(vectors.shape)) * 100:.1f}%")
else:
    print(f"⚠️  WARNING: Still dense ({vectors.nbytes / 1e9:.2f} GB)")
    print("⚠️  Run the preprocessing script!")

print("=" * 60)

# Pydantic Schemas
class RecommendRequest(BaseModel):
    movie: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

class RecommendationItem(BaseModel):
    title: str
    tmdb_id: int | None

class RecommendResponse(BaseModel):
    input_movie: str
    recommendations: List[RecommendationItem]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    memory_format: str
    num_movies: int

# Routes
@app.get("/")
def root():
    return {"status": "Hybrid recommender (memory optimized)"}

@app.get("/health", response_model=HealthResponse)
def health_check():
    mem_format = "sparse" if scipy.sparse.issparse(vectors) else "dense"
    return HealthResponse(
        status="ok", 
        model_loaded=True,
        memory_format=mem_format,
        num_movies=len(new_df)
    )

@app.get("/movies", response_model=List[str])
def get_movies():
    return sorted(new_df["title"].unique().tolist())

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        # Find movie
        df_match = new_df[new_df["title"] == req.movie]
        if df_match.empty:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        movie_index = df_match.index[0]
        
        # Compute similarity (works for both sparse and dense)
        sims = linear_kernel(vectors[movie_index], vectors).flatten()
        
        # Use argpartition for memory efficiency
        k = min(req.top_k + 1, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
        
        # Remove input movie (first result)
        top_indices = top_indices[1:req.top_k + 1]
        
        # Build response
        recommendations = []
        for i in top_indices:
            title = new_df.iloc[i]["title"]
            tmdb_id = new_df.iloc[i]["id"]
            recommendations.append(
                RecommendationItem(title=title, tmdb_id=tmdb_id)
            )
        
        return RecommendResponse(
            input_movie=req.movie, 
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))