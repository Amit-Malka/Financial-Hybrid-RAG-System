# Financial RAG System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A sophisticated Retrieval-Augmented Generation (RAG) system designed for analyzing SEC 10-Q quarterly financial reports. This system provides intelligent question-answering capabilities over complex financial documents, combining multiple retrieval strategies with advanced language models.

## ⚠️ Important Disclaimers

**FINANCIAL INFORMATION WARNING**: This system is designed for research and educational purposes only. It should not be used as a substitute for professional financial advice, investment recommendations, or legal counsel.

- **Not Financial Advice**: The information provided by this system is for informational purposes only and should not be considered as financial, investment, or legal advice.
- **No Warranties**: This software is provided "as is" without any warranties, express or implied, including but not limited to warranties of merchantability or fitness for a particular purpose.
- **SEC Compliance**: This tool does not guarantee compliance with SEC regulations or requirements. Users are responsible for ensuring their own compliance with applicable laws and regulations.
- **Data Accuracy**: While efforts are made to provide accurate information, the system may not detect all errors or inconsistencies in financial documents.
- **Professional Consultation**: Always consult with qualified financial professionals, accountants, and legal experts for financial decision-making.

## 🚀 Features

### Core Capabilities
- **Hybrid Retrieval System**: Combines dense embeddings, sparse TF-IDF, and graph-based retrieval for optimal results
- **SEC 10-Q Processing**: Specialized handling of quarterly financial reports with semantic understanding
- **Multi-Modal Analysis**: Processes text, tables, and structured financial data
- **Advanced Question Answering**: Provides context-aware responses with source citations
- **Real-time Evaluation**: Integrated RAGAS and DeepEval metrics for quality assessment

### Technical Features
- **Metadata-Driven Architecture**: 5-field metadata schema optimized for financial documents
- **Graph Database Integration**: Neo4j-powered document structure understanding
- **Modular Tool System**: Specialized tools for tables, MD&A, risk factors, and general queries
- **User-Configurable Parameters**: Adjustable retrieval weights and optional reranking
- **Comprehensive UI**: Gradio-based interface with multiple specialized tabs

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Router    │───▶│   Tool System   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Hybrid Retriever│    │  Document Store  │    │   LLM Engine    │
│ • Dense (70%)   │◀──▶│  • ChromaDB      │    │ • Gemini 2.5     │
│ • TF-IDF (30%)  │    │  • Neo4j Graph   │    │ • Context Window │
│ • Graph (+15%)  │    └──────────────────┘    └─────────────────┘
└─────────────────┘
```

### Technology Stack
- **Backend**: Python 3.10+
- **Vector Database**: ChromaDB with HuggingFace embeddings
- **Graph Database**: Neo4j for document structure
- **LLM**: Google Gemini 2.5 Flash
- **Document Processing**: LlamaIndex + SEC-parser
- **UI Framework**: Gradio
- **Orchestration**: LangChain
- **Evaluation**: RAGAS, DeepEval

## 📋 Prerequisites

Before running this application, ensure you have:

- Python 3.10 or higher
- Google Cloud API key (for Gemini)
- Neo4j database (optional, for graph features)
- At least 8GB RAM recommended
- 10GB free disk space for models and data

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/financial-rag-system.git
cd financial-rag-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
Create a `.env` file in the project root:
```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional (for enhanced features)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
COHERE_API_KEY=your_cohere_key  # For reranking
```

## 🚀 Quick Start

### Basic Usage
```python
from src.ui.gradio_app import create_interface

# Launch the web interface
interface = create_interface()
interface.launch()
```

### Command Line Usage
```bash
# Start the application
python -m src.ui.gradio_app

# Run tests
python -m pytest tests/

# Run evaluation
python -m src.evaluation.ragas_evaluation
```

## 📖 Usage Guide

### 1. Document Upload
- Upload SEC 10-Q PDF files through the web interface
- The system automatically processes and chunks the document
- Metadata is extracted and stored for enhanced retrieval

### 2. Query Interface
- Use the **Q&A** tab for general questions
- **Table Queries** tab for financial data questions
- **Financial Analysis** tab for comprehensive assessment
- **Summary** tab for document overview

### 3. Configuration
- Adjust retrieval weights in the **Configuration** tab
- Toggle optional reranking for improved accuracy
- Monitor system performance in **System Info**

### 4. Evaluation
- Use the **Evaluation** tab to assess system performance
- RAGAS metrics provide quality scores
- Ground truth comparison available

## 🔧 Configuration

### Retrieval Parameters
```python
# Default settings (configurable via UI)
DENSE_WEIGHT = 0.7          # Semantic search weight
TFIDF_WEIGHT = 0.3          # Keyword search weight
GRAPH_ENHANCEMENT = 0.15    # Graph boost weight
CHUNK_SIZE = 400            # Text chunk size
CHUNK_OVERLAP = 50          # Chunk overlap
```

### Supported File Types
- PDF documents (SEC 10-Q reports)
- HTML exports from SEC EDGAR
- Structured financial data formats

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4-core processor
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Recommended Requirements
- **CPU**: 8-core processor or higher
- **RAM**: 16GB or more
- **Storage**: 20GB SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, for faster processing)

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test category
pytest tests/test_retrieval.py
```

## 🤝 Contributing

We welcome contributions! For development setup:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](docs/LICENSE) file for details.

## 🙏 Acknowledgments

- **SEC EDGAR** for providing public financial data
- **Google** for Gemini language models
- **LlamaIndex** for document processing framework
- **LangChain** for orchestration capabilities
- **Neo4j** for graph database technology

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/yourusername/financial-rag-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/financial-rag-system/discussions)
- **Documentation**: [Technical Documentation](docs/CODEMAP.md)

---

**Disclaimer**: This software is for educational and research purposes. Not intended for production financial analysis or investment decision-making. Always consult qualified financial professionals for investment advice.
