import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: str, retriever=None) -> None:
        try: 
            self.session_id = session_id

            # load LLM and prompts once
            self.llm = self._load_llm()
            self.contextualize_prompt : ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt : ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            # lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized successfully", session_id=session_id)


        except Exception as e: raise DocumentPortalException("Error initializing ConversationalRAG", sys)
    # PUBLIC API

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[dict[str, Any]] = None,
    ):
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path, 
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True # only if you trust the index
                )
            
            if search_kwargs is None:
                search_kwargs = {"k": k}
            
            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            self._build_lcel_chain()

            log.info("FAISS retriever load surccesfully",
                index_path=index_path,
                index_name=index_name,
                k = k,
                session_id=self.session_id,
                )
            return self.retriever
            
        except Exception as e:
            log.error("Error loading retriever from FAISS", error=str(e))
            raise DocumentPortalException("Error loading retriever from FAISS", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        try:
            if self.chain is None:
                raise DocumentPortalException("RAG Chain is not initialized. Call load_retriever_from_faiss() before invoke().", sys)
            
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(
                payload,
            )

            if not answer:
                log.warning("No answer received", session_id=self.session_id)
                
            log.info(
                "Chain invoked succesfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=answer[:50]
                )
            return answer
        except Exception as e:
            log.error("Error invoking ConversationalRAG", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Error invoking ConversationalRAG", sys)

    # internals

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM not loaded")
            log.info("LLM loaded succefully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Error loading LLM via ModelLoader", error=str(e))
            raise DocumentPortalException("Error loading LLM", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No Retriever set before building chain", sys)
        
            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # 2) Retrieve relevant documents
            retrieve_docs = question_rewriter | self.retriever | self._format_docs
            
            # 3) Answer the question
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history")
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
            log.info("LCEL chain built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Error building LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Error building LCEL chain", e)