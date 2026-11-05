from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode
import os
import datetime
import json
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import base64, io, os, sys
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup


from dotenv import load_dotenv
load_dotenv(".env")

"""
We are defining some tools
We will also define some more tools that our agents will use in the future
"""

tavily_tool = TavilySearch(max_results=5)
repl = (
    PythonREPL()
)  # Warning: This executes code locally, which can be unsafe when not sandboxed


# Default vector database
DEFAULT_FILES_DIR = "./default_files"
DEFAULT_DB_DIR = "./default_vector_db"
MANIFEST_PATH = os.path.join(DEFAULT_DB_DIR, "manifest.json")


def _default_files_manifest():
    """Fingerprint (name, size, mtime) of every PDF in ./default_files."""
    if not os.path.isdir(DEFAULT_FILES_DIR):
        return {}
    manifest = {}
    for name in sorted(os.listdir(DEFAULT_FILES_DIR)):
        if name.lower().endswith(".pdf"):
            st = os.stat(os.path.join(DEFAULT_FILES_DIR, name))
            manifest[name] = [st.st_size, int(st.st_mtime)]
    return manifest


def sync_default_vector_db(embeddings=None):
    """Keep the default knowledge base in sync with ./default_files.

    Replaces the old Pathway VectorStoreServer: instead of a separate streaming
    server watching a folder, the FAISS index is rebuilt whenever a PDF in
    ./default_files is added, removed or modified (checked on startup and before
    each retrieve_documents call).
    """
    manifest = _default_files_manifest()
    try:
        with open(MANIFEST_PATH) as f:
            if json.load(f) == manifest and manifest:
                return
    except (OSError, ValueError):
        pass

    if not manifest:
        if os.path.isdir(DEFAULT_DB_DIR):
            shutil.rmtree(DEFAULT_DB_DIR)
        return

    try:
        print(f"Indexing {len(manifest)} PDF(s) from {DEFAULT_FILES_DIR}...")
        documents = []
        for name in manifest:
            documents.extend(PyPDFLoader(os.path.join(DEFAULT_FILES_DIR, name)).load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        db = FAISS.from_documents(docs, embeddings or OpenAIEmbeddings())
        db.save_local(DEFAULT_DB_DIR)
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f)
        print(f"Default knowledge base ready ({len(docs)} chunks).")
    except Exception as e:
        print(f"Error building default vector DB: {e}")


sync_default_vector_db()


@tool
def duckduckgo_search(query: Annotated[str, "The search query."]):
    """Use this tool to perform a search using DuckDuckGo and return the content of the search results."""
    search_tool = DuckDuckGoSearchResults(output_format="list")
    results = search_tool.invoke(query)
    contents = []

    for result in results:
        url = result["link"]
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            print(f"Retrieving content from: {url}")
            response = requests.get(
                url, timeout=10, headers=headers
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            contents.append(
                {
                    "url": url,
                    "content": text[
                        :1000
                    ],  # Limit content to first 1000 characters for brevity
                }
            )
        except requests.RequestException as e:
            print(f"Failed to retrieve content: {e}")
            # contents.append({
            #     'url': url,
            #     'content': f"Failed to retrieve content: {e}"
            # })

    return "Search results:\n" + "\n".join(
        [f"{content['url']}\n{content['content']}\n" for content in contents]
    )


@tool
def python_repl(
    code: Annotated[
        str, "The python code to execute. Don't show the plot in the output."
    ],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""

    try:
        # Execute the code
        result = repl.run(code)

        # Check if a plot was generated
        if plt.get_fignums():
            # Ensure the generated_files directory exists
            os.makedirs("generated_files", exist_ok=True)
            # Save the plot to a file in the generated_files directory
            file_path = os.path.join("generated_files", "plot.png")
            plt.savefig(file_path)
            plt.close()
            result_str = f"Successfully executed:\n```python\n{code}\n```\nPlot saved to: {file_path}"
        else:
            result_str = (
                f"Successfully executed:\n```python\n{code}\n```\nOutput: {result}"
            )

    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

    return result_str


@tool
def retrieve_documents(query: str):
    """Searches for documents related to a given query."""
    sync_default_vector_db()
    try:
        db_dir = DEFAULT_DB_DIR if os.path.isdir(DEFAULT_DB_DIR) else "./vector_db"
        if not os.path.isdir(db_dir):
            return "No document database is available yet."
        db = FAISS.load_local(
            db_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever()        
        if retriever is None:
            return "Retriever is not initialized."
        try:
            docs = retriever.invoke(query)
            return [doc.page_content for doc in docs]
        except Exception as e:
            return f"Failed to retrieve documents. Error: {repr(e)}"
    except Exception as e:
        return f"Failed to load the vector database. Error: {repr(e)}"


@tool
def user_file_retriever_tool(query: str):
    """Searches for documents related to a given query on a user file."""
    if not os.path.isdir("./vector_db"):
        return "No user file has been uploaded yet."
    db = FAISS.load_local("./vector_db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = db.as_retriever()    
    if retriever is None:
        return "Retriever is not initialized."
    try:
        docs = retriever.invoke(query)
        return [doc.page_content for doc in docs]
    except Exception as e:
        return f"Failed to retrieve documents. Error: {repr(e)}"


@tool
def date_tool():
    """This tool will return the current date and time."""

    return f"The current date and time is {datetime.datetime.now()}"


tools = [
    tavily_tool,
    python_repl,
    retrieve_documents,
    date_tool,
    user_file_retriever_tool,
    duckduckgo_search,
]
tool_node = ToolNode(tools)
