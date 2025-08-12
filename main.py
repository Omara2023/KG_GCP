from fastapi import FastAPI, Depends
from dependencies import get_orchestrator 
from orchestrators.base import BaseRAGOrchestrator
from models.user_query import UserQuery
from models.terminal_llm_response import TerminalLLMResponse

app = FastAPI()

@app.post("/query", response_model=TerminalLLMResponse)
async def service_query(query: UserQuery, orchestrator: BaseRAGOrchestrator = Depends(get_orchestrator)):
    return await orchestrator.generate(query.text)
    
         

