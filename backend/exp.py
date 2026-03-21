"""
PROPER preprocessing - Converts dense arrays to sparse and removes unused objects
Run this ONCE before starting your server
"""

import joblib
import scipy.sparse
import numpy as np
import os
import gc

MODEL_PATH = "model/hybrid_recommender.pkl"
BACKUP_PATH = "model/hybrid_recommender_backup.pkl"

print("=" * 60)
print("STEP 1: Loading original model...")
print("=" * 60)

bundle = joblib.load(MODEL_PATH)

# Show what we have
print("\n📦 Original bundle contents:")
for key in bundle.keys():
    print(f"  - {key}")

# Backup original (only once)
if not os.path.exists(BACKUP_PATH):
    print("\n💾 Creating backup...")
    joblib.dump(bundle, BACKUP_PATH)
    print(f"✅ Backup saved to {BACKUP_PATH}")
else:
    print(f"\n✅ Backup already exists at {BACKUP_PATH}")

print("\n" + "=" * 60)
print("STEP 2: Optimizing vectors...")
print("=" * 60)

vectors = bundle["vectors"]
print(f"Before: {vectors.nbytes / 1e9:.2f} GB (dtype: {vectors.dtype})")

# Convert to sparse CSR + float32
vectors_sparse = scipy.sparse.csr_matrix(vectors, dtype=np.float32)
print(f"After:  {vectors_sparse.data.nbytes / 1e6:.2f} MB (sparse CSR, dtype: {vectors_sparse.dtype})")
print(f"Savings: {(vectors.nbytes - vectors_sparse.data.nbytes) / 1e9:.2f} GB")

# CRITICAL: Delete the old dense array to free memory
del vectors
gc.collect()

print("\n" + "=" * 60)
print("STEP 3: Removing unnecessary objects...")
print("=" * 60)

# We only need these 3 things for the API:
optimized_bundle = {
    "new_df": bundle["new_df"],      # Movie metadata
    "vectors": vectors_sparse,        # Sparse vectors (was 5.45 GB, now 22 MB)
}

# OPTIONAL: Keep similarity_scores if you use it (you don't in current code)
# If you DO need it, convert it to sparse too:
if "similarity_scores" in bundle:
    sim_scores = bundle["similarity_scores"]
    print(f"similarity_scores: {sim_scores.nbytes / 1e6:.2f} MB")
    
    # Check if it's worth making sparse
    sparsity = (1 - np.count_nonzero(sim_scores) / np.prod(sim_scores.shape)) * 100
    if sparsity > 50:
        print(f"  Converting to sparse (sparsity: {sparsity:.1f}%)...")
        optimized_bundle["similarity_scores"] = scipy.sparse.csr_matrix(sim_scores, dtype=np.float32)
    else:
        print(f"  Keeping dense (only {sparsity:.1f}% sparse)")
        optimized_bundle["similarity_scores"] = sim_scores.astype(np.float32)

# Remove old bundle from memory
del bundle
gc.collect()

print("\n✅ Removed unused objects:")
print("  - pt_title_to_idx (not used in API)")
print("  - title_to_rating (not used in API)")
print("  - id_to_votes (not used in API)")
print("  - title_to_id (not used in API)")

print("\n" + "=" * 60)
print("STEP 4: Saving optimized model...")
print("=" * 60)

joblib.dump(optimized_bundle, MODEL_PATH)

print(f"\n✅ Optimized model saved to {MODEL_PATH}")

# Show final size
file_size_mb = os.path.getsize(MODEL_PATH) / 1e6
print(f"📊 File size: {file_size_mb:.2f} MB")

print("\n" + "=" * 60)
print("DONE! Now restart your FastAPI server")
print("=" * 60)
print("\nExpected RAM usage:")
print("  - Before: ~5.5 GB ❌")
print("  - After:  ~50 MB ✅")