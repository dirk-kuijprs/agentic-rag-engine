import os
import yaml
import logging
from typing import List, Union
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from sentence_transformers import CrossEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgenticRAGEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.embeddings = OpenAIEmbeddings(model=self.config["embeddings"]["model"])
        self.llm = ChatOpenAI(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"]
        )
        self.persist_directory = self.config.get("vector_store", {}).get("persist_directory", "./chroma_db")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.agent_executor = None

    def ingest_documents(self, data_dir: str):
        logger.info(f"Ingesting documents from {data_dir}...")
        loader = DirectoryLoader(data_dir, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["ingestion"]["chunk_size"],
            chunk_overlap=self.config["ingestion"]["chunk_overlap"]
        )
        texts = text_splitter.split_documents(documents)
        
        self.vector_store = Chroma.from_documents(
            texts, 
            self.embeddings, 
            persist_directory=self.persist_directory
        )
        logger.info(f"Vector store created and persisted at {self.persist_directory}")

    def rerank_documents(self, query: str, docs: List):
        if not docs:
            return []
        
        doc_contents = [d.page_content for d in docs]
        pairs = [[query, content] for content in doc_contents]
        scores = self.reranker.predict(pairs)
        
        # Sort docs by score
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for score, d in scored_docs[:5]] # Return top 5

    def self_rag_logic(self, query: str) -> str:
        """Self-RAG logic: Retrieve -> Rerank -> Evaluate -> Refine."""
        logger.info(f"Self-RAG: Processing query '{query}'")
        
        # 1. Retrieve
        initial_docs = self.vector_store.similarity_search(query, k=10)
        
        # 2. Rerank
        reranked_docs = self.rerank_documents(query, initial_docs)
        context = "\n\n".join([d.page_content for d in reranked_docs])
        
        # 3. Evaluate and Generate (simplified Self-RAG loop)
        prompt = f"System: Evaluate the context for relevance to the query. If relevant, synthesize a response. If not, state that more information is needed.\nQuery: {query}\nContext: {context}"
        response = self.llm.invoke(prompt)
        
        return response.content

    def setup_agent(self):
        tools = [
            Tool(
                name="knowledge_base",
                func=self.self_rag_logic,
                description="Use this tool to retrieve and synthesize information using Self-RAG logic."
            )
        ]

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        logger.info("Agent setup complete with Self-RAG tools.")

    def query(self, user_input: str):
        if not self.agent_executor:
            self.setup_agent()
        return self.agent_executor.invoke({"input": user_input})

if __name__ == "__main__":
    # Example usage
    # engine = AgenticRAGEngine()
    # engine.ingest_documents("./data")
    # response = engine.query("What are the key findings in the latest report?")
    # print(response["output"])
    pass
