import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# Load dataset
df = pd.read_csv("data.csv", on_bad_lines='warn')
df['combined'] = df["title"].astype(str) + " by " + df["authors"].astype(str)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create embeddings
combined_embeddings = model.encode(df['combined'].astype(str).tolist(), convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(combined_embeddings.shape[1])
index.add(combined_embeddings)

# Save index and optionally the DataFrame
faiss.write_index(index, "combined.index")
df.to_pickle("data.pkl")
