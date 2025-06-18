import os
from flask import Flask, request, jsonify, Response
from gemini_gen_module import GeminiCaller
from vertex_rag_module import VertexAICaller

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_query():
    """Handles incoming POST requests with a user query, performs RAG, and returns the grounded answer."""
    print("Entered backend handler successfully.")

    if not request.is_json:
        print("Request not json apparently.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        print("No user query apparently.")
        return jsonify({"error": "Missing 'query' in request"}), 400

    print(f"Received query from frontend: '{user_query}'")

    vertex_ai_caller = VertexAICaller()
    contexts = vertex_ai_caller.run_vertex_ai_search(user_query)
    print("DEBUG: Returning pickled SearchResponse to frontend.", flush=True)
    if isinstance(contexts, bytes): # Check if it's indeed pickled bytes
        print("DEBUG: Returning pickled SearchResponse to frontend.", flush=True)
        # Send as raw binary data, setting appropriate content type
        return Response(contexts, mimetype='application/octet-stream')
    else:
        # Fallback if something went wrong and it's not bytes (e.g., error from run_vertex_ai_search)
        print("ERROR: Expected pickled response but got something else.", flush=True)
        return jsonify({"error": "Failed to retrieve raw debug response."}), 500

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
