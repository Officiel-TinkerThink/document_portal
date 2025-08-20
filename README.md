# Document Portal

A comprehensive document processing and analysis platform that enables users to interact with documents in multiple ways. Built with FastAPI and modern AI/ML technologies, this tool provides powerful document processing capabilities through an intuitive web interface.

## Key Features

### 1. Document Analysis
- Extract and analyze metadata from documents
- Generate summaries and key insights
- Identify important entities and topics

### 2. Document Comparison
- Compare two documents side by side
- Highlight differences and changes
- Track modifications between versions
- Generate detailed comparison reports

### 3. Document Chat (RAG)
- Chat with your documents using AI
- Get instant answers from document content
- Support for multiple documents in conversation
- Context-aware responses based on document content
- Support for multiple document upload and queries

## Technology Stack
- **Backend**: FastAPI
- **Frontend**: HTML5, CSS3, JavaScript
- **AI/ML**: LangChain, Embedding Models
- **Vector Database**: FAISS
- **Document Processing**: PyPDF, python-docx

## Getting Started

### Create Project Folder and Environment Setup

```bash
# Create a new project folder
mkdir <project_folder_name>

# Move into the project folder
cd <project_folder_name>

# Open the folder in VS Code
code .

# Create a new Conda environment with Python 3.10
conda create -p <env_name> python=3.10 -y

# Activate the environment (use full path to the environment)
conda activate <path_of_the_env>

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### Minimum Requirements for the Project

#### LLM Models
- **Groq** (Free)
- **OpenAI** (Paid)
- **Gemini** (15 Days Free Access)
- **Claude** (Paid)
- **Hugging Face** (Free)
- **Ollama** (Local Setup)

#### Embedding Models
- **OpenAI**
- **Hugging Face**
- **Gemini**

#### Vector Databases
- **In-Memory**
- **On-Disk**
- **Cloud-Based**

### API Keys

#### GROQ API Key
- [Get your API Key](https://console.groq.com/keys)  
- [Groq Documentation](https://console.groq.com/docs/overview)

#### Gemini API Key
- [Get your API Key](https://aistudio.google.com/apikey)  
- [Gemini Documentation](https://ai.google.dev/gemini-api/docs/models)