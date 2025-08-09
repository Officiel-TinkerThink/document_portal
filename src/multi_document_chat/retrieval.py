import sys
import os
from operator import itemgetter
from typing import Optional

from langchain_core.messages import BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader
from model.models import PromptType
from prompt.prompt_library import PROMPT_REGISTRY

class ConversationalRAG:
    def __init__(self, session_id: str, retriever=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            if retriever is None:
                raise ValueError("Retriever is required")
            self.retriever = retriever
            self._build_lcel_chain()
            self.log.info("ConversationalRAG initialized", session_id=self.session_id)

        except Exception as e:
            self.log.error("Error initializing ConversationalRAG", error=str(e))
            raise DocumentPortalException("Error initializing ConversationalRAG", sys)

    def load_retriever_from_faiss(self, index_path: str):
        """
        Load a FAISS vectorstore from disk and convert to retriever
        """
        try:
            embeddings = ModelLoader().load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index not found: {index_path}")

            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True # only if you trust the index
            )
            self.retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 5})
            self.log.info("Loaded retriever from FAISS index", faiss_path=index_path, session_id=self.session_id)

            return self.retriever
            
        except Exception as e:
            self.log.error("Error loading retriever from FAISS", error=str(e))
            raise DocumentPortalException("Error loading retriever from FAISS", sys)

    def invoke(self, user_input: str, chat_history: Optional[list[BaseMessage]] = None) -> str:
        try:
            chat_history = chat_history or []
            payload={"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                self.log.warning("No answer received", user_input=user_input, session_id=self.session_id)
                return "No answer generated"
            self.log.info(
                "Chain invoked successfully", 
                user_input=user_input, 
                session_id=self.session_id, 
                answer_preview=answer[:50])
            return answer
        except Exception as e:
            self.log.error("Error invoking ConversationalRAG", error=str(e))
            raise DocumentPortalException("Error invoking ConversationalRAG", sys)

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM is required")
            self.log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            self.log.error("Error loading LLM", error=str(e))
            raise DocumentPortalException("Error loading LLM", sys)

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def _build_lcel_chain(self):
        try:
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            self.chain = (
                {
                    "context": retrieve_docs,
                    "question": itemgetter("input"),
                    "chat_history": itemgetter("chat_history") 
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            self.log.info("LCEL chain built successfully", session_id=self.session_id)
            return self.chain
        except Exception as e:
            self.log.error("Error building LCEL chain", error=str(e))
            raise DocumentPortalException("Error building LCEL chain", sys)




