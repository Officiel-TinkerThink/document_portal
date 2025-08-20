import sys
from dotenv import load_dotenv
import pandas as pd
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from model.models import SummaryResponse, PromptType
from prompt.prompt_library import PROMPT_REGISTRY #type: ignore
from utils.model_loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
import reprlib

class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()

        # Prepare parsers
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | self.parser
        log.info("DocumentComparatorLLM initialized with model and parser.")
    def compare_documents(self, combined_docs: str):
        """
        Compare two documents  and returns a structured comparison.
        """
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions()
            }
            log.info("Starting document comparison", inputs=reprlib.repr(inputs))
            response = self.chain.invoke(inputs)
            log.info("Document comparison completed", response=reprlib.repr(response))
            return self._format_response(response)
        except Exception as e:
            log.error("Failed to compare documents", error=str(e))
            raise DocumentPortalException("Failed to compare documents", sys)

    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame: #type: ignore
        """
        Format the response from the LLM into a structured format
        """
        try:
            df = pd.DataFrame(response_parsed)
            log.info("Response formatted into DataFrame", dataframe=reprlib.repr(str(df)))
            return df

        except Exception as e:
            log.error("Error formatting response into DataFrame", error=str(e))
            raise DocumentPortalException("Error formatting response", sys)

    
