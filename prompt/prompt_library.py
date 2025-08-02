from langchain_core.prompts import ChatPromptTemplate

# Prepare prompt template
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant traned to analyze and summarize documents
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

document_comparison_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant traned to compare and summarize documents
Return ONLY valid JSON matching the exact schema below.



The Input Document:
{document_text}

Your response should follow the following format:
{format_instructions}

""")

PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt
}