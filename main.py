import os
from flask import Flask, request, jsonify
from .gemini_gen_module import GeminiCaller
from .vertex_rag_module import VertexAICaller

app = Flask(__name__)

SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID", "your-search-app-id")

@app.route('/query', methods=['POST'])
def ask():
    """
    Handles incoming POST requests with a user query, performs RAG,
    and returns the grounded answer.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "Missing 'query' in request"}), 400

    print(f"Received query from frontend: '{user_query}'")

    # Module 1: Retrieve context
    vertex_ai_caller = VertexAICaller()
    contexts = vertex_ai_caller.run_vertex_ai_search(user_query)

    gemini_caller = GeminiCaller()
    if not contexts:
        # If no context is found, you can still try to answer or inform the user
        print("No relevant contexts found from Vertex AI Search. Attempting to answer without grounding.")
        final_response = gemini_caller.generate_response(user_query, [])
    else:
        final_response = gemini_caller.generate_response(user_query, contexts)

    return jsonify(final_response)

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
