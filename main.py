import os
import logging
from flask import Flask, request, jsonify
from logging_config import setup_logging

app_logger = setup_logging()
app = Flask(__name__)
main_file_logger = logging.getLogger(__name__)

from gemini_gen_module import GeminiCaller
from vertex_rag_module import VertexAICaller

main_file_logger.info("Flask app starting up.")

@app.route('/query', methods=['POST'])
def handle_query():
    """Handles incoming POST requests with a user query, performs RAG, and returns the grounded answer."""

    if not request.is_json:
        main_file_logger.info("Request not isn't JSON.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        main_file_logger.info("No user query included.")
        return jsonify({"error": "Missing 'query' in request"}), 400

    main_file_logger.info(f"Received query from frontend: '{user_query}'")

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
