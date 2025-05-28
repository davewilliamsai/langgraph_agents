import os
import requests
from datetime import datetime
from langchain_core.tools import tool
from langchain_community.utilities import ArxivAPIWrapper

API_KEY = os.getenv("SERP_API_KEY")

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from Process mining literature.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in th Process Mining documents"

    result = []
    for i, doc in enumerate(docs):
        result.append(f"Document {i+1}:\n{doc.page_content}, Metadata:\n{doc.metadata}")

    return "\n\n".join(result)

@tool
def google_scholar_tool(query: str) -> str:
    """
    This tool searches and returns information about scientific research papers with google scholar.
    """

    url = f"https://google.serper.dev/scholar?q={query}&apiKey={API_KEY}"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)

    return response.text

@tool
def arxiv_tool(query: str) -> str:
    """
    This tool searches and returns information about scientific research papers on Arxiv.
    """
    arxiv = ArxivAPIWrapper(top_k_results=10)
    docs = arxiv.run(query)
    return docs

@tool
def save_tool(content: str, filename: str) -> str:
    """
    Save the provided content to a text file.

    Args:
        content: The text content to save.
        filename: The name of the output file (without .txt extension).
    """
    date_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not filename.endswith('.txt'):
        filename = f"./docs/saves/{date_now}_{filename}.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Document saved to '{filename}'."
    except Exception as e:
        return f"Error saving document: {str(e)}"