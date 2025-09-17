from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import faiss
import asyncio
import requests
import os
import json
from sentence_transformers import SentenceTransformer
from dotenv import dotenv_values


OLLAMA_API =  os.environ['OLLAMA_API']
OLLAMA_MODEL =  os.environ['OLLAMA_MODEL']

# Load local embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load prebuilt index
combined_index = faiss.read_index("combined.index")
df = pd.read_pickle("data.pkl")

app = Flask(__name__)
CORS(app)

def faiss_lookup(query:str, k=10):
    print(k, query, flush=True)
    
    print(k, query, flush=True)
    query_vec=model.encode([query],convert_to_numpy=True)
    D, I = combined_index.search(query_vec, k=25)
    results = []
    for index, location in enumerate(I[0]):
        row=df.iloc[location]
        row = {
            "title": row["title"],
            "authors": row["authors"],
            "average_rating": row["average_rating"]
        }
        print(row, flush=True)
        
        results.append(row)
        if len(results) >= 2 and len(results) >=k:
            break
    return results

def llm_call(messages, tools, stream):
    llm_response = requests.post(OLLAMA_API, headers= { "Content-Type": "application/json" },
			json= {
				"model": OLLAMA_MODEL,
                "messages": messages,
                "tools": tools,
                "stream": stream,
                })
    data = llm_response.json()
    print(data, flush=True)
    assistant_msg = data["message"]
    messages.append(assistant_msg)
    
    if "tool_calls" in assistant_msg:
        print("Book_Search called", flush=True)
        print(assistant_msg, flush=True)
        for call in assistant_msg["tool_calls"]:
            if call["function"]["name"] == "book_search":
                if call["function"]["arguments"]["query"]:
                    args = call["function"]["arguments"]
                    tool_result = faiss_lookup(args["query"], args["numberOfBooks"])
                    # Append tool output to the chat history
                    tool_output= set({})
                    for result in tool_result:
                        tool_output.add(result["title"] +" by " +result["authors"] +" with a rating of " + str(result["average_rating"].item()))
                    print(tool_output)
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_name": "book_search"
                    })
                    return llm_call(messages,[tools[1]],stream)
            if call["function"]["arguments"]["reply"]:
                return call["function"]["arguments"]["reply"]
    else:
        print(assistant_msg, flush=True)
        reply = assistant_msg["content"]
        if "</think>" in reply:
            end_index = reply.index("</think>")
            reply = reply[end_index+9:]
            print(reply, flush=True)
        return reply

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search/llm/combined", methods=["POST"])
def search_llm_combined():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    messages = [
        {'role': 'system', 'content': "/no_think You are a bookstore chatbot, you will be given the user's question. You can use the book_search tool to find books if the user asks about it. ONLY USE book_search WHEN NEEDED. Only give the user the number of books they ask for. Keep responses as short as possible. If a user is looking for a similar book, do not return to the user a book with the same title."},
        {'role': 'user', 'content': query},
    ]
    
    tools = [{
        "type": "function",
            "function": {
                "name": "book_search",
                "description": "Searches the bookstore database for book titles and authors using semantic search on 'query'. Returns a list of books in no particular order. Choose only the most applicable ones. Request more than one for similarity searches so it doesn't return a the original book. ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search terms that should be used to find a book for the user."
                        },
                        "numberOfBooks": {
                            "type": "integer",
                            "description": "The number of books that you want returned"
                        }
                    },
                    "required": ["query"]
                }
            }
    }, {
        "type": "function",
            "function": {
                "name": "reply",
                "description": "Sends the imput to the user as a reply to their question. Use if you do not need any other tools.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reply": {
                            "type": "string",
                            "description": "Reply to send to the user."
                        }
                    },
                    "required": ["query"]
                }
            }
    }]
    
    # 3. Get LLM response
    assistant_msg = llm_call(messages,tools,False)
        
    # else:
    return jsonify({
        "query": query,
        "llm_response": assistant_msg,
    })

    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)