import os
import logging

from flask import Flask, request, jsonify
from logging_config import setup_logging
from gemini_caller import GeminiCaller
from vertex_ai_caller import VertexAICaller

app_logger = setup_logging()
main_file_logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
VERTEX_AI_SEARCH_LOCATION  = os.environ.get("GCP_VERTEX_LOCATION")
GEMINI_LLM_LOCATION  = os.environ.get("GCP_GEMINI_LOCATION") 
ENGINE_ID = os.environ.get("GCP_ENGINE_ID")   
GEMINI_MODEL_NAME = "gemini-1.5-flash"

app = Flask(__name__)

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

    main_file_logger.info(f"User query: '{user_query}'")

    vertex_ai_caller = VertexAICaller(PROJECT_ID, VERTEX_AI_SEARCH_LOCATION, ENGINE_ID)
    contexts = vertex_ai_caller.run_vertex_ai_search(user_query)
  
    gemini_caller = GeminiCaller(GEMINI_MODEL_NAME)
    if not contexts:
        main_file_logger.info("No relevant contexts found from Vertex AI Search. Attempting to answer without grounding.")
        final_response = gemini_caller.generate_response(user_query, [])
    else:
        final_response = gemini_caller.generate_response(user_query, contexts)

    return jsonify(final_response)

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
