# Quick Start Guide

## Prerequisites

Before getting started, ensure you have:

- **Python**: Version 3.10 or higher
- **API Keys**:
  - Google API Key (required for Gemini LLM)
  - Cohere API Key (optional, for reranking)
  - Neo4j credentials (optional, for graph features)
- **System Requirements**: At least 8GB RAM, 10GB free disk space

## Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-rag-system.git
cd financial-rag-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
# Required: Google API Key for Gemini
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Cohere API Key for reranking
COHERE_API_KEY=your_cohere_api_key_here

# Optional: Neo4j for graph features
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

## Running the Application

### Start the Web Interface

```bash
# From project root with virtual environment activated
python -m src.ui.gradio_app
```

The application will start and provide a local URL (typically `http://localhost:7860`).

## Using the Application

### 1. Basic Q&A Workflow

1. **Navigate to Q&A Tab**: Open the main interface in your browser
2. **Configure API Keys**: Enter your Google API key in the interface
3. **Upload Document**: Upload a SEC 10-Q PDF file (under 50 pages recommended)
4. **Process Document**: Click "Process File" to analyze and index the document
5. **Ask Questions**: Enter questions about the financial report
6. **Optional Features**:
   - Toggle reranking for improved accuracy (requires Cohere API key)
   - Add to graph database for enhanced retrieval (requires Neo4j)

### 2. Specialized Features

- **Table Queries**: Use the dedicated tab for questions about financial tables and numeric data
- **Financial Analysis**: Get comprehensive analysis of financial health and risk factors
- **Document Summary**: Generate executive summaries of the entire report
- **System Configuration**: Adjust retrieval weights and system parameters

### 3. Evaluation

Use the Evaluation tab to assess system performance:
- Enter test questions and ground truth answers
- View RAGAS metrics (Context Precision, Recall, Faithfulness)
- Compare system responses against expected answers

## Testing

Run the test suite to verify everything is working:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_retrieval.py
```

## Troubleshooting

### Common Issues

**Virtual Environment Activation Issues (Windows)**:
```powershell
# If activation is blocked, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**API Key Errors**:
- Ensure your Google API key has Gemini API access
- Check that API keys are properly set in environment variables or the UI

**Memory Issues**:
- Close other applications if you encounter memory errors
- Consider using smaller documents (under 30 pages) for testing

**Neo4j Connection Issues**:
- Ensure Neo4j is running and accessible
- Verify connection credentials in the UI

### Getting Help

- Check the [main README](../README.md) for detailed documentation
- Review [technical documentation](CODEMAP.md) for implementation details
- Open an issue on GitHub for bugs or feature requests

---

**Remember**: This tool is for educational and research purposes. Always consult financial professionals for investment decisions.


