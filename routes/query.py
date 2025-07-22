import logging 
from flask import Blueprint, request, jsonify
from flask.views import MethodView
from services.vertex_service import VertexRagService
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
query_bp = Blueprint("query", __name__)

class QueryView(MethodView):
    def post(self):
        """Handles incoming POST requests with a user query, performs RAG, and returns the grounded answer."""
        if not request.is_json:
            logger.info("Request not isn't JSON.")
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        user_query = data.get('query')
        if not user_query:
            logger.info("No user query included.")
            return jsonify({"error": "Missing 'query' in request"}), 400

        logger.info(f"User query: '{user_query}'")

        vertex_service = VertexRagService() 
        if (contexts := vertex_service.search(user_query)):
            logger.info("Successfully retrieved contexts from RAG engine.")
        else:
            logger.info("Failed to retrieve contexts from RAG engine.")     
        
        gemini_service = GeminiService()
        response = gemini_service.respond(user_query, contexts)

        return jsonify(response), 200
    
query_bp.add_url_rule('/query', view_func=QueryView.as_view("query_post"))
