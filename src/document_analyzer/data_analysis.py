import os
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
import sys

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from prompt.prompt_library import *


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

            self.prompt = prompt

            self.log.info("DocumentAnalyzer initialized successfully")

        except Exception as e:
            self.log.error("Failed to initialize DocumentAnalyzer: {e}")
            raise DocumentPortalException("Failed to initialize DocumentAnalyzer", sys)

    def analyze_document(self, document_path):
        try:
            model = self.model_loader.load_llm()
            analysis_result = model.analyze(document_path)
            return self.fixing_parser.invoke(analysis_result)
        except Exception as e:
            self.log.error("Failed to analyze document", error=str(e))
            raise DocumentPortalException("Failed to analyze document", sys)
        