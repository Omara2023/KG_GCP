import os
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, Part
from vertexai.preview.rag import RagResource, VertexRagStore
from flask import Flask, request, jsonify

app = Flask(__name__)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("GCP_REGION")
VERTEX_AI_SEARCH_DATA_STORE_RESOURCE_NAME = os.environ.get("VERTEX_AI_SEARCH_DATA_STORE_RESOURCE_NAME")

vertexai.init(project=PROJECT_ID, location=REGION)

model = GenerativeModel("gemini-1.5-flash-001")


@app.route("/query", methods=["POST"])
def handle_query():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        print(f"Received query: '{user_query}'")
        print(f"Using Data Store: {VERTEX_AI_SEARCH_DATA_STORE_RESOURCE_NAME}")

        # Define the retrieval tool, pointing to your Vertex AI Search Data Store
        retrieval_tool = Tool.from_retrieval(
            retrieval=VertexRagStore(
                rag_resources=[
                    RagResource(
                        rag_corpus=VERTEX_AI_SEARCH_DATA_STORE_RESOURCE_NAME,
                        # Optional: use_recitation_check=True (can improve answer quality)
                        # Optional: source_max_tokens=2000 (limit context length to save tokens)
                    )
                ],
                # Optional: Configure how many chunks to retrieve (default is usually 5)
                # rag_retrieval_config=RagRetrievalConfig(top_k=5)
            )
        )

        # Make the generative content call to Gemini, providing the grounding tool
        response = model.generate_content(
            user_query,
            tools=[retrieval_tool]
        )

        # --- Parse Gemini's response ---
        generated_text = ""
        for part in response.candidates[0].content.parts:
            if isinstance(part, Part) and part.text:
                generated_text += part.text

        citations = []
        if response.candidates[0].grounding_metadata:
            # Iterate through grounded_contents to find source IDs
            # The structure can be a bit nested
            for gr in response.candidates[0].grounding_metadata.retrieval_tool_code.grounded_contents:
                if gr.segments:
                    for segment in gr.segments:
                        # The source_id will typically point to your GCS file name or part of it
                        if segment.source_id:
                            citations.append(segment.source_id)

        # Remove duplicate citations and format nicely
        unique_citations = list(set(citations))
        formatted_citations = []
        for citation in unique_citations:
            # Often source_id will be the gs://bucket/path/to/file.pdf. Extract just the file name.
            file_name = os.path.basename(citation.replace('gs://', ''))
            formatted_citations.append(file_name)


        print("Response generated successfully.")
        return jsonify({
            "response": generated_text,
            "citations": formatted_citations
        })

    except Exception as e:
        app.logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e), "details": str(e)}), 500

if __name__ == "__main__":
    # For local testing only. Cloud Run will use 'gunicorn' or similar.
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))