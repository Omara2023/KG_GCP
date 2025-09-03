from clients.vertex_retrieval_client import VertexRetrievalClient
from clients.gemini_client import GeminiClient
from agentic.answer_critic import AnswerCritic

system_instructions = (
    "You are an expert assistant. Use the retrieved contexts provided to answer the user's question as faithfully as possible. "
    "Do not invent facts or speculate. If the contexts contain relevant information, use them. "
    "If the contexts do not include enough information to answer the question fully, you may rely on your own general knowledge to fill in the gaps, "
    "but clearly indicate when this is the case. If the question cannot be answered with the context or your own knowledge, say so."
)

async def run(query: str, retriever: VertexRetrievalClient, llm_client: GeminiClient, answer_critic: AnswerCritic, n: int = 1) -> str:
    """Choose retreival strategy, run context retreival, judge answer, iterate if needed and return."""
    llm_client.system_instruction = system_instructions
    while (n >= 0):
        contexts = [c.text for c in retriever.run_context_retrieval(query)]
        prompt = _format_prompt(query, contexts)
        output = llm_client.prompt(prompt)
        judegemnt = answer_critic.judge(query, output)

        if judegemnt["verdict"] == "satisfactory":
            break
        else:
            if ((next_query := judegemnt.get("follow_up")) is not None):
                query = next_query
                n -= 1
            else:
                raise ValueError("Judgement dictionary lacks a follow_up query.")
    return output

def _format_prompt(query: str, contexts: list[str]) -> str:
    prompt_parts = ["\n\n--- Retrieved Contexts ---"]
    for i, context in enumerate(contexts):
        prompt_parts.append(f"\nContext {i+1}:\nText:{context}")
    prompt_parts.append("\n-------------------------\n")
    prompt_parts.append("Using the above contexts, answer the following question as accurately as possible.")
    prompt_parts.append(f"\n\nUser's Question: {query}")
    return "\n".join(prompt_parts)


