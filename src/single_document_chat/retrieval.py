import sys
import os
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader, PromptType
from prompt.prompt_library import PROMPT_REGISTRY




class ConversationalRAG:
    def __init__(self, session_id: str, retriever) -> None:
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.retriever = retriever
        except Exception as e:
            self.log.error("Error initializing ConversationalRAG", error=str(e))
            raise DocumentPortalException("Error initializing ConversationalRAG", sys)
    
    def _load_llm(self):
        try:
            pass
        except Exception as e:
            self.log.error("Error loading LLM via ModelLoader", error=str(e))
            raise DocumentPortalException("Error loading LLM", sys)
    
    def _get_session_history(self, session_id: str):
        try:
            pass
        except Exception as e:
            self.log.error("Error getting session history", session_id=session_id,error=str(e))
            raise DocumentPortalException("Error getting session history", sys)

    def load_retriever_from_faiss(self):
        try:
            pass
        except Exception as e:
            self.log.error("Error loading retriever from FAISS", error=str(e))
            raise DocumentPortalException("Error loading retriever from FAISS", sys)

    def invoke(self):
        try:
            pass
        except Exception as e:
            self.log.error("Error invoking ConversationalRAG", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Error invoking ConversationalRAG", sys)