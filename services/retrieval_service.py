from clients.vertex_retrieval_client import VertexRetrievalClient
from clients.gemini_client import GeminiClient

async def single_pass(query: str, retriever: VertexRetrievalClient, llm_client: GeminiClient) -> str:
    llm_client.system_instruction = [
        "You are an expert assistant. Use the retrieved contexts provided to answer the user's question as faithfully as possible. "
        "Do not invent facts or speculate. If the contexts contain relevant information, use them. "
        "If the contexts do not include enough information to answer the question fully, you may rely on your own general knowledge to fill in the gaps, "
        "but clearly indicate when this is the case. If the question cannot be answered with the context or your own knowledge, say so."
    ]
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


