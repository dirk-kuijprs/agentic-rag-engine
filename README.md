# Agentic RAG Engine 🤖📚

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://github.com/langchain-ai/langchain)

Advanced Retrieval-Augmented Generation (RAG) system integrated with ReAct agents for complex information retrieval and multi-step reasoning. Built with LangChain and FAISS, this engine is designed for production-scale document intelligence.

## 🌟 Key Features
- **Agentic Reasoning**: Uses ReAct (Reason + Act) patterns to intelligently decide when to retrieve information.
- **Hybrid Retrieval**: Combines dense vector search with metadata filtering (via FAISS).
- **Scalable Ingestion**: Asynchronous document processing with `DirectoryLoader` and recursive splitting.
- **Configurable Pipeline**: YAML-driven configuration for LLMs, embeddings, and retrieval parameters.
- **Modular Design**: Easy to extend with new tools (e.g., Web Search, SQL, API executors).

## 🛠️ Installation

```bash
git clone https://github.com/dirk-kuijprs/agentic-rag-engine.git
cd agentic-rag-engine
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```
2. Prepare your data: Place PDF documents in the `./data` directory.
3. Run the engine:
   ```python
   from rag_agent import AgenticRAGEngine
   
   engine = AgenticRAGEngine(config_path="config.yaml")
   engine.ingest_documents("./data")
   response = engine.query("Synthesize the main arguments from the financial reports.")
   print(response["output"])
   ```

## 👨‍💻 Author
**Dirk Kuijprs**  
Data Scientist at G42

Special thanks to **Muhammad Ajmal Siddiqui** for his mentorship and guidance. Connect with him on [LinkedIn](https://www.linkedin.com/in/muhammadajmalsiddiqi/).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
