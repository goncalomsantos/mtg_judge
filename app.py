import os
import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter

load_dotenv()
parent_path = Path(__file__).parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = Chroma(
    collection_name="mtg-rulling-new",
    embedding_function=embedding_model,
    persist_directory="db",
)


system_prompt = f"""
    You are a Magic: The Gathering judge and you have to answers judge calls of games or answer questions related to MTG cards and gameplay.

    Instruction:
    Answer the provided user query based on the provided Context.
    If the answer for the question is not on the provided Context, check if it's on the chat history, if it's answer is based on that.
    Strictly only rely on the sources provided in generating your response. Never rely on external sources.
    Always answer in english.
    If the user asks the question in another language that is not english, say that you only talk in english.
"""
messages = [("system", system_prompt)]


def loading_phase():
    # phase 1: load (pre-RAG)

    # step 1.1: document load
    print("loading rulings to json loader...")
    file_path = parent_path / "file_sources" / "rulings.json"

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        text_content=False,
    )
    documents = loader.load()
    docs_sample = documents[9000:9100]
    print("done")

    # step 1.2: document transformation
    # not needed as the json is already splitted into documents
    # here we would split the documents into chunks

    # step 1.3: embedding 1:1

    # texts = []
    # metadata = []
    # for doc in documents:
    #    texts.append(doc.page_content)
    #    metadata.append(doc.metadata)
    # embeddings = embedding_model.embed_documents(texts)
    print("embedding models...")
    texts = [doc.page_content for doc in docs_sample]
    # embeddings = embedding_model.embed_documents(texts)
    print("done")

    print("adding documents to collection...")
    vector_db.add_documents(docs_sample)
    print("done")

    return True


def predict(message, history):

    # Phase 2: RAG generation
    # Step 2.1: Embed user query
    user_query = message

    embeddings_user_query = embedding_model.embed_query(user_query)

    # Step 2.2: similarity search
    relevant_chunks = vector_db.similarity_search(user_query)

    print(relevant_chunks)

    relevant_chunks_text = ""
    for relevant_text in relevant_chunks:
        relevant_chunks_text += relevant_text.page_content + "\n"

    chat_history = ""

    for index, messages in enumerate(history):
        chat_history += f"""User query {index}:{messages[0]} Assistant Response {index}: {messages[1]}"""

    # Step 2.4 create prompt
    prompt = f"""
        User query:
        {user_query}

        Context:
        {relevant_chunks_text}

        Chat History:
        {chat_history}
    """

    # Step 2.5: call LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    messages = [("human", prompt)]

    llm_response = llm.invoke(messages)

    return llm_response.content


loading_phase()
gr.ChatInterface(predict).launch(debug=True)
