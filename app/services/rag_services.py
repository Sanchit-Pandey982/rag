from phase1 import RAGSystem

from app.schemas.chat import ChatRequest


class RAGService:

    def __init__(
        self,
        rag: RAGSystem
    ):
        self.rag = rag


    def run_once(
        self,
        request: ChatRequest
    ) -> dict:

        chat_history = [
            message.model_dump()
            for message in request.chat_history
        ]

        result = self.rag.run_once(
            raw_query=request.raw_query,
            user_id=request.user_id,
            chat_history=chat_history,
            k=request.k,
            rewrite_query=request.rewrite_query,
            distance_threshold=request.distance_threshold
        )

        return result