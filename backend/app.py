from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import faiss
import asyncio
import requests
import os
import json
from sentence_transformers import SentenceTransformer
from dotenv import dotenv_values


OLLAMA_API =  os.environ['OLLAMA_API']
OLLAMA_MODEL =  os.environ['OLLAMA_MODEL']

print(OLLAMA_API)
# Load local embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load your dataset
df = pd.read_csv("data.csv", on_bad_lines='warn', delimiter=',')

# Create a combined column for easier reference if needed
df['combined'] = df["title"].astype(str) + 'by ' + df['authors'].astype(str)

# Embed dataset
title_embeddings = model.encode(df['title'].astype(str).tolist(), convert_to_numpy=True)
authors_embeddings = model.encode(df['authors'].astype(str).tolist(), convert_to_numpy=True)
combined_embeddings = model.encode(df['combined'].astype(str).tolist(), convert_to_numpy=True)

# Build FAISS index for titles
title_dimension = title_embeddings.shape[1]
title_index = faiss.IndexFlatL2(title_dimension)
title_index.add(title_embeddings)

# Build FAISS index for authors
authors_dimension = authors_embeddings.shape[1]
authors_index = faiss.IndexFlatL2(authors_dimension)
authors_index.add(authors_embeddings)

# Build FAISS index for combined
combined_dimension = combined_embeddings.shape[1]
combined_index = faiss.IndexFlatL2(combined_dimension)
combined_index.add(combined_embeddings)

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search/llm/title", methods=["POST"])
def search_llm_title():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # 1. FAISS search
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = title_index.search(query_vec, k=5)
    results = [df.iloc[i] for i in I[0]]

    # 2. Build prompt for LLM
    prompt_text = f"User asked: '{query}'\nTop matches:\n"
    for r in results:
        prompt_text += f"- {r['title']} by {r['authors']} (Rating: {r['average_rating']})\n"
    prompt_text += "\nAnswer the user's question using these search results for your information. Only suggest the number of books requested by the user."

    # 3. Get LLM response
    llm_response = requests.post(OLLAMA_API, headers= { "Content-Type": "application/json" },
			json= {
				"model": "gemma3:4b",
				"prompt": prompt_text,
                "stream": False
                })
    if llm_response.status_code != 200:
        return jsonify({
            "error": f"Ollama returned {llm_response.status_code}",
            "details": llm_response.text
        }), 500

    # 4. Parse Ollama response JSON
    try:
        data = llm_response.json()
        answer_text = data.get("response", "")
    except Exception:
        answer_text = llm_response.text  # fallback

    return jsonify({
        "query": query,
        "matches": [
            {
                "title": r["title"],
                "authors": r["authors"],
                "average_rating": r["average_rating"]
            }
            for r in results
        ],
        "llm_answer": answer_text
    })
    
    # axios.get(fetch(OLLAMA_API, {
	# 		method: "POST",
	# 		headers: { "Content-Type": "application/json" },
	# 		body: JSON.stringify({
	# 			model: "llama3",
	# 			prompt: userPrompt,
	# 			stream: false,
	# 		}),
	# 	});

    # return jsonify({"answer": answer_text})
    
@app.route("/search/llm/combined", methods=["POST"])
def search_llm_combined():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # 1. FAISS search
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = combined_index.search(query_vec, k=25)
    results = [df.iloc[i] for i in I[0]]

    # 2. Build prompt for LLM
    prompt_text = f"You are a bookstore chatbot, you will be given the user's question as well as semantic search results through the database of books that the bookstore has. Use this information to best answer the User's question. User asked: '{query}'\nTop matches:\n"
    for r in results:
        prompt_text += f"- {r['title']} by {r['authors']} (Rating: {r['average_rating']})\n"
    prompt_text += "\nFollow the user's prompt to the best of your ability. Your reply will be fed directly to the user, please reply accordingly. Ignore the search results entirely if they dont ask about books."
    print(prompt_text)
    # 3. Get LLM response
    llm_response = requests.post(OLLAMA_API, headers= { "Content-Type": "application/json" },
			json= {
				"model": "gemma3:4b",
				"prompt": prompt_text,
                "stream": False
                })
    if llm_response.status_code != 200:
        return jsonify({
            "error": f"Ollama returned {llm_response.status_code}",
            "details": llm_response.text
        }), 500

    # 4. Parse Ollama response JSON
    try:
        data = llm_response.json()
        answer_text = data.get("response", "")
    except Exception:
        answer_text = llm_response.text  # fallback

    return jsonify({
        "query": query,
        "matches": [
            {
                "title": r["title"],
                "authors": r["authors"],
                "average_rating": r["average_rating"]
            }
            for r in results
        ],
        "llm_answer": answer_text
    })
    
    # axios.get(fetch(OLLAMA_API, {
	# 		method: "POST",
	# 		headers: { "Content-Type": "application/json" },
	# 		body: JSON.stringify({
	# 			model: "llama3",
	# 			prompt: userPrompt,
	# 			stream: false,
	# 		}),
	# 	});

    # return jsonify({"answer": answer_text})

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
