import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain_core.documents import Document
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

vector_db_comp_rules = Chroma(
    collection_name="mtg-comp-rules-new",
    embedding_function=embedding_model,
    persist_directory="db_comp_rules",
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


def load_rulings() -> List[Document]:
    """Load and return ruling chunks from the JSON file."""
    print("loading rulings to json loader...")
    file_path: Path = parent_path / "file_sources" / "rulings.json"
    loader: JSONLoader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        text_content=False,
    )
    documents: List[Document] = loader.load()
    print("done")
    return documents


def decode_json_page_content(doc: Document) -> Document:
    try:
        data = json.loads(doc.page_content)
        # Use the 'comment' field for content and 'oracle_id' for metadata
        comment = data.get("comment", "")
        oracle_id = data.get("oracle_id", None)
        metadata = {"oracle_id": oracle_id} if oracle_id is not None else {}
        return Document(page_content=comment, metadata=metadata)
    except Exception:
        # If not JSON, return as is
        return doc


def add_chunks_to_vector_db(
    chunks: List[Document], vector_db: Chroma = vector_db
) -> None:
    """Add a list of Document chunks to the vector database in batches of 5461."""
    print("embedding models...")
    texts: List[str] = [doc.page_content for doc in chunks]
    print("done")
    print("adding documents to collection in batches...")
    max_batch_size = 5461
    total = len(chunks)
    num_batches = (total + max_batch_size - 1) // max_batch_size
    for i in range(0, total, max_batch_size):
        batch_num = i // max_batch_size + 1
        batch = chunks[i : i + max_batch_size]
        print(
            f"Adding batch {batch_num} out of {num_batches} ({len(batch)} documents)..."
        )
        vector_db.add_documents(batch)
    print("done")


def predict(
    message: str, history: List[Tuple[str, str]], db: Chroma = vector_db
) -> str:

    # Phase 2: RAG generation
    # Step 2.1: Embed user query
    user_query = message

    # Step 2.2: similarity search
    relevant_chunks = db.similarity_search(user_query, 20)

    print(relevant_chunks)

    relevant_chunks_text = ""
    for relevant_text in relevant_chunks:
        relevant_chunks_text += relevant_text.page_content + "\n"

    chat_history_chunks = ""
    # TODO: check if messages global var is being overwritten
    for index, chunk in enumerate(history):
        chat_history_chunks += (
            f"""User query {index}:{chunk[0]} Assistant Response {index}: {chunk[1]}"""
        )

    # Step 2.4 create prompt
    prompt = f"""
        User query:
        {user_query}

        Context:
        {relevant_chunks_text}

        Chat History:
        {chat_history_chunks}
    """

    # Step 2.5: call LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    global messages
    messages = messages + [("human", prompt)]
    breakpoint()
    llm_response = llm.invoke(messages)

    return llm_response.content


def predict_comp_rules(message: str, history: List[Tuple[str, str]]) -> str:
    return predict(message, history, vector_db_comp_rules)


def load_and_chunk_magic_rules_txt() -> List[Document]:
    file_path = parent_path / "file_sources" / "MagicCompRules.txt"
    loader = TextLoader(str(file_path), encoding="utf-8")
    raw_docs = loader.load()
    text = raw_docs[0].page_content

    # Find the character offset for line 180 (where the real rules start)
    lines = text.splitlines(keepends=True)
    start_offset = sum(len(line) for line in lines[:180])

    # Find all Glossary and Credits section offsets
    glossary_matches = list(re.finditer(r"^Glossary$", text, re.MULTILINE))
    credits_matches = list(re.finditer(r"^Credits$", text, re.MULTILINE))
    # Use the second occurrence of 'Glossary' as the real glossary section start
    glossary_offset = glossary_matches[1].start() if len(glossary_matches) > 1 else None
    credits_offset = credits_matches[1].start() if len(credits_matches) > 1 else None

    chunks = []

    # 1. Chunk main rules (from start_offset to glossary_offset)
    # Find all section headers (e.g., 100. General)
    section_header_pattern = re.compile(r"^(\d{3}\. .+)$", re.MULTILINE)
    rule_pattern = re.compile(r"^(\d{3}\.[\da-z]+\.)", re.MULTILINE)

    # Find all section headers and their positions
    section_headers = list(
        section_header_pattern.finditer(text, start_offset, glossary_offset)
    )
    for idx, section_match in enumerate(section_headers):
        section_header = section_match.group(1).strip()
        section_start = section_match.end()
        section_end = (
            section_headers[idx + 1].start()
            if idx + 1 < len(section_headers)
            else glossary_offset
        )
        section_text = text[section_start:section_end]
        # Find all rules in this section
        rule_matches = list(rule_pattern.finditer(section_text))
        for r_idx, rule_match in enumerate(rule_matches):
            rule_number = rule_match.group(1).strip()
            rule_start = rule_match.end()
            rule_end = (
                rule_matches[r_idx + 1].start()
                if r_idx + 1 < len(rule_matches)
                else len(section_text)
            )
            rule_body = section_text[rule_start:rule_end].strip()
            if rule_body:
                page_content = f"{rule_number} {rule_body}"
                chunks.append(
                    Document(
                        page_content=page_content, metadata={"section": section_header}
                    )
                )

    # 2. Chunk glossary (from glossary_offset to credits_offset) by term
    if glossary_offset and credits_offset:
        glossary_text = text[glossary_offset:credits_offset]
        # Each term is a line in Title Case, not indented, followed by its definition(s)
        term_pattern = re.compile(r"^([A-Z][A-Za-z0-9 ',\-\(\)]+)$", re.MULTILINE)
        matches_glossary = list(term_pattern.finditer(glossary_text))
        # Skip the first match if it's 'Glossary'
        start_idx = (
            1
            if matches_glossary and matches_glossary[0].group(1).strip() == "Glossary"
            else 0
        )
        for i in range(start_idx, len(matches_glossary)):
            match_glossary = matches_glossary[i]
            term = match_glossary.group(1).strip()
            start_glossary = match_glossary.end()
            end_glossary = (
                matches_glossary[i + 1].start()
                if i + 1 < len(matches_glossary)
                else len(glossary_text)
            )
            definition = glossary_text[start_glossary:end_glossary].strip()
            if definition:
                chunks.append(
                    Document(page_content=definition, metadata={"glossary_term": term})
                )
    return chunks


# TODO: Uncomment this when you want to load the comp rules and rulings
# TODO: make the load changeable in the terminal
# chunks_comp_rules = load_and_chunk_magic_rules_txt()
# chunks_rulings = load_rulings()
#
# chunks_rulings_decoded = [decode_json_page_content(chunk) for chunk in chunks_rulings]
#
# all_chunks = chunks_comp_rules + chunks_rulings_decoded


# add_chunks_to_vector_db(chunks_comp_rules, vector_db_comp_rules)

# start_time = time.time()
# add_chunks_to_vector_db(all_chunks)
# end_time = time.time()
# print(f"Time taken to add all chunks to vector DB: {end_time - start_time:.2f} seconds")


gr.ChatInterface(predict).launch(debug=True)
# gr.ChatInterface(predict_comp_rules).launch(debug=True)
