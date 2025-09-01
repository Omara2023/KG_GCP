from clients.vertex_retrieval_client import VertexRetrievalClient
from clients.gemini_client import GeminiClient

async def single_pass(query: str, retriever: VertexRetrievalClient, llm_client: GeminiClient) -> str:
    contexts = [c.text for c in retriever.run_context_retrieval(query)]
    prompt = _format_prompt(query, contexts)
    return llm_client.prompt(prompt)

def _format_prompt(query: str, contexts: list[str]) -> str:
    prompt_parts = ["\n\n--- Retrieved Contexts ---"]
    for i, context in enumerate(contexts):
        prompt_parts.append(f"\nContext {i+1}:\nText:{context}")
    prompt_parts.append("\n-------------------------\n")
    prompt_parts.append("Using the above contexts, answer the following question as accurately as possible.")
    prompt_parts.append(f"\n\nUser's Question: {query}")
    return "\n".join(prompt_parts)


