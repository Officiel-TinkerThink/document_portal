import os
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
import sys

class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization.
    """

    def __init__(self):
        self.model_loader: ModelLoader = ModelLoader()
        self.logger: CustomLogger = CustomLogger()
        self.output_parser: JsonOutputParser = JsonOutputParser()
        self.fixing_parser: OutputFixingParser = OutputFixingParser.from_output_parser(self.output_parser)

    def analyze_document(self, document_path):
        try:
            model = self.model_loader.load_llm()
            analysis_result = model.analyze(document_path)
            return self.fixing_parser.invoke(analysis_result)
        except Exception as e:
            self.logger.error("Failed to analyze document", error=str(e))
            raise DocumentPortalException("Failed to analyze document", sys)
        