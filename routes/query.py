import logging 
from flask import Blueprint, request, jsonify
from flask.views import MethodView
from services.vertex_service import VertexService
from services.gemini_service import GeminiService

main_file_logger = logging.getLogger(__name__)
query_bp = Blueprint("query", __name__)

class QueryView(MethodView):
    def post(self):
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

        vertex_service = VertexService()
        contexts = vertex_service.search(user_query)

        gemini_service = GeminiService()
        response = gemini_service.respond(user_query, contexts)

        return jsonify(response)
    
query_bp.add_url_rule('/query', view_func=QueryView.as_view("query_post"))
