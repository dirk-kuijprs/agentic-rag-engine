import os
import yaml
import logging
from typing import List, Union
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

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
        self.vector_store = None
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
        
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        logger.info("Vector store created successfully.")

    def setup_agent(self):
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Run ingest_documents first.")

        retriever = self.vector_store.as_retriever()
        
        def retrieval_tool_func(query: str) -> str:
            docs = retriever.get_relevant_documents(query)
            return "\n\n".join([d.page_content for d in docs])

        tools = [
            Tool(
                name="knowledge_base",
                func=retrieval_tool_func,
                description="Use this tool to retrieve information from the internal knowledge base."
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
        logger.info("ReAct agent setup complete.")

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
