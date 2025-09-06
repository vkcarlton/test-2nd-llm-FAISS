from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Load local embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load your dataset
df = pd.read_csv("data.csv", on_bad_lines='warn', delimiter=',')

# Create a combined column for easier reference if needed
df['combined'] = df["title"].astype(str) + ', ' + df['authors'].astype(str) + ', ' + df['average_rating'].astype(str)

# Embed dataset
title_embeddings = model.encode(df['title'].astype(str).tolist(), convert_to_numpy=True)
authors_embeddings = model.encode(df['authors'].astype(str).tolist(), convert_to_numpy=True)

# Build FAISS index for titles
title_dimension = title_embeddings.shape[1]
title_index = faiss.IndexFlatL2(title_dimension)
title_index.add(title_embeddings)

# Build FAISS index for authors
authors_dimension = authors_embeddings.shape[1]
authors_index = faiss.IndexFlatL2(authors_dimension)
authors_index.add(authors_embeddings)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search/title", methods=["POST"])
def search_title():
    query = request.json.get("query")
    print(request)
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Encode query locally
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = title_index.search(query_vec, k=10)

    # Return full rows as dicts
    results = []
    for idx in I[0]:
        row = df.iloc[idx]
        results.append({
            "title": row["title"],
            "authors": row["authors"],
            "average_rating": row["average_rating"]
        })
        print(results)
    return jsonify(results)

@app.route("/search/authors", methods=["POST"])
def search_authors():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Encode query locally
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = authors_index.search(query_vec, k=10)

    # Return full rows as dicts
    results = []
    for idx in I[0]:
        row = df.iloc[idx]
        results.append({
            "title": row["title"],
            "authors": row["authors"],
            "average_rating": row["average_rating"]
        })
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
