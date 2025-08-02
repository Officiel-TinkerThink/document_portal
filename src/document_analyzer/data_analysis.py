import os
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
import sys

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from prompt.prompt_library import PROMPT_REGISTRY


class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization.
    """

    def __init__(self):
        self.log: CustomLogger = CustomLogger().get_logger(__name__)
        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            # Prepare parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

            self.prompt = PROMPT_REGISTRY["document_analysis"]

            self.log.info("DocumentAnalyzer initialized successfully")
        except Exception as e:
            self.log.error("Failed to initialize DocumentAnalyzer: {e}")
            raise DocumentPortalException("Failed to initialize DocumentAnalyzer", sys)

    def analyze_document(self, document_text: str) -> dict:
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            
            self.log.info("Meta-data analysis chain intialized")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            self.log.info("Metadata extraction successful", keys=list(response.keys()))

            return response

        except Exception as e:
            self.log.error("Failed to analyze document", error=str(e))
            raise DocumentPortalException("Failed to analyze document", sys)
        