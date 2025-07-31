from langchain_core.prompts import ChatPromptTemplate

# Prepare prompt template
prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant traned to analyze and summarize documents
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")