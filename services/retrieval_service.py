import logging
from clients.vertex_retrieval_client import VertexRetrievalClient
from clients.gemini_client import GeminiClient
from agentic.answer_critic import AnswerCritic
from models.retrieved_context import RetrievedContext

system_instructions = (
    "You are an expert assistant. Use the retrieved contexts provided to answer the user's question as faithfully as possible. "
    "Do not invent facts or speculate. If the contexts contain relevant information, use them. "
    "If the contexts do not include enough information to answer the question fully, you may rely on your own general knowledge to fill in the gaps, "
    "but clearly indicate when this is the case. If the question cannot be answered with the context or your own knowledge, say so."
)

async def run(query: str, retriever: VertexRetrievalClient, llm_client: GeminiClient, answer_critic: AnswerCritic, n: int = 1) -> str:
    """Choose retreival strategy, run context retreival, judge answer, iterate if needed and return."""
    logger = logging.getLogger(__name__)
    count = 0
    llm_client.system_instruction = system_instructions
    while (n > 0):
        logger.info(f"Query: {query}")
        count += 1
        contexts = [c for c in retriever.run_context_retrieval(query)]
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
    logger.info(f"Recursed {count} times.")
    return output

def _format_prompt(query: str, contexts: list[RetrievedContext]) -> str:
    prompt_parts = ["\n\n--- Retrieved Contexts ---"]
    for i, context in enumerate(contexts):
        prompt_parts.append(f"\nContext {i+1}:\nText: {context.text}\nSource: {context.source_file}")
    prompt_parts.append("\n-------------------------\n")
    prompt_parts.append("Using the above contexts, answer the following question as accurately as possible. Enumerate the name of the contexts used in your response before you actual answer.")
    prompt_parts.append(f"\n\nUser's Question: {query}")
    prompt_parts.append("""Present yout answer in such a way with name referring to source (gcp location removed from file name rendering say a book name.): 
                        Context 1: (name of context used)
                        Context 2: (name of context used)
                        Context n: (name of context used)
                        
                        <The answer, referring to which context a given point is drawn from.>""")
    return "\n".join(prompt_parts)


