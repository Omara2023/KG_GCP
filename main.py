import os
import logging
import vertexai
from flask import Flask, request, jsonify
from logging_config import setup_logging

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") # Your Google Cloud project ID.
LOCATION = os.environ.get("GCP_LOCATION") # The region of your search engine.
ENGINE_ID = os.environ.get("GCP_ENGINE_ID") # The ID of your Vertex AI Search app (engine).         

app_logger = setup_logging()
main_file_logger = logging.getLogger(__name__)

from gemini_gen_module import GeminiCaller
from vertex_rag_module import VertexAICaller

try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except Exception as e:
    main_file_logger.error(f"Error initializing Vertex AI: {e}")
else:
    main_file_logger.info(f"Vertex AI initialized for project {PROJECT_ID} in location {LOCATION}")

app = Flask(__name__)

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

    vertex_ai_caller = VertexAICaller(PROJECT_ID, LOCATION, ENGINE_ID)
    contexts = vertex_ai_caller.run_vertex_ai_search(user_query)

    gemini_caller = GeminiCaller()
    if not contexts:
        main_file_logger.info("No relevant contexts found from Vertex AI Search. Attempting to answer without grounding.")
        final_response = gemini_caller.generate_response(user_query, [])
    else:
        final_response = gemini_caller.generate_response(user_query, contexts)

    return jsonify(final_response)

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
