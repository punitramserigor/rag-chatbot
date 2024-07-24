from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_milvus import Milvus
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import PromptTemplate
import os
import json


class RAG:
    def __init__(self, session_id) -> None:
        self.expr = f"session_id == '{session_id}'"
        self.session_id = session_id
        self.memory_file = f"{session_id}_memory.json"
        self.embedding_function = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "",
            ],
        )
        self.connection_args = {
            "uri": os.getenv("ZILLIZ_CLOUD_URI"),
            "username": os.getenv("ZILLIZ_CLOUD_USERNAME"),
            "password": os.getenv("ZILLIZ_CLOUD_PASSWORD"),
            "token": os.getenv("ZILLIZ_CLOUD_API_KEY"),
            "secure": True,
        }
        self.prompt_template = """You are an AI Assistant having a conversation with a human.

            You will be given the extracted parts of a long document as a context, and a question related to that context.

            Use only and only the provided context to answer the question. DO NOT use your own knowledge base to answer the questions.

            ### Context:
            {context}

            ### Previous Conversation:
            {chat_history}

            ### Query:
            Human: {human_input}
            AI:"""
        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "context"],
            template=self.prompt_template,
        )
        
        self.chat_memory = FileChatMessageHistory(
            file_path=self.memory_file
        )

        self.chain = load_qa_chain(
            self.llm, chain_type="stuff", prompt=prompt
        )

    def load_conversation(self, file_path):
        with open(file_path, "r") as file:
            conversation = json.load(file)
            
        # Get the latest 10 messages
        latest_messages = conversation[-10:]

        formatted_output = []
        for message in latest_messages:
            message_type = message["type"]
            content = message["data"]["content"]
            formatted_output.append(f"{message_type}: {content}")
        
        return "\n".join(formatted_output)

    def load_db(self, file_path: str):
        success = self.ingest_documents(file_path=file_path)
        return success

    def ingest_documents(self, file_path):
        if ".pdf" in file_path:
            self.loader = PyPDFLoader(file_path)
        if ".md" in file_path:
            self.loader = UnstructuredMarkdownLoader(file_path)
        if ".txt" in file_path:
            self.loader = TextLoader(file_path)
        if ".json" in file_path:
            self.loader = JSONLoader(file_path)
        if ".docx" in file_path:
            self.loader = Docx2txtLoader(file_path)
        self.document_list = self.loader.load_and_split(
            text_splitter=self.text_splitter
        )
        for document in self.document_list:
            document.metadata = {"session_id": self.session_id}

        # Connect to Zilliz Cloud and create the vector store
        self.vector_store = Milvus.from_documents(
            documents=self.document_list,
            embedding=self.embedding_function,
            connection_args=self.connection_args
        )
        return True

    def invoke(self, query: str):
        self.vector_store = Milvus(
            embedding_function=self.embedding_function,
            connection_args=self.connection_args,
        )
        retriever = self.vector_store.as_retriever(
            search_type="similarity"
        )
        retrieved_docs = retriever.invoke(query)
        conversation = self.load_conversation(self.memory_file)
        self.response = self.chain.invoke(
            {
                "input_documents": retrieved_docs,
                "human_input": query,
                "chat_history": conversation,
            },
            return_only_outputs=True,
        )
        self.chat_memory.add_user_message(query)
        self.chat_memory.add_ai_message(self.response["output_text"])
        return self.response["output_text"]

    def delete_db(self):
        self.vector_store = Milvus(
            embedding_function=self.embedding_function,
            connection_args=self.connection_args,
        )
        self.vector_store.delete(expr=self.expr)

        # Delete the memory file
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
