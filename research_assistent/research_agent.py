import os
from typing import Annotated, Sequence, TypedDict, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage,SystemMessage, ToolMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from database_document_setup import db_loader, personal_info

import requests
from datetime import datetime
from langchain_core.tools import tool
from langchain_community.utilities import ArxivAPIWrapper

load_dotenv()
API_KEY = os.getenv("SERP_API_KEY")

llm = ChatOpenAI(
    model= "gpt-4o", temperature=0
)

embeddings = OpenAIEmbeddings(
    model= "text-embedding-3-small"
)

pdf_path = os.getcwd() + "/docs_pm4py/literature"

file_paths = [os.path.join(pdf_path, f) for f in os.listdir(pdf_path)]

# Database loader
retriever = db_loader(pdf_path, file_paths, embeddings)

personal_info = personal_info()

############## Setting up tools ##################

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


tools = [retriever_tool,
         google_scholar_tool,
         arxiv_tool,
         save_tool]

llm = llm.bind_tools(tools)

########### Setting up Agents #################

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    last_ai_content: Optional[str]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

topic = "Process Mining"

system_prompt_call_llm = f"""
You are a highly intelligent AI assistant specializing in Process Mining. Your core task is to answer user questions and support scientific research using all available tools and documents in your knowledge base.

🔐 Saving Content Instructions:
- If the user asks to **save content**, use the `save_tool`.
- Only call the tool when:
  - You know what content to save (your last response in last_ai_content).
  - You have a file name (ask the user if it's missing).
- If no filename is given, generate one based on the topic or question. Keep it short (< 20 characters), lowercase, no spaces.


🧠 Core Responsibilities for Research:
Use all tools available:
- `retriever_tool` – for querying internal PDF documents.
- `arxiv_tool` – for accessing scientific papers from arXiv.
- `google_scholar_tool` – for scientific papers from Google Scholar.
You are encouraged to make **multiple tool calls** as needed to ensure completeness. Always aim to gather as much **relevant, high-quality information** as possible.

📄 Document-Based Question Answering:
Use the retriever_tool to look up content in internal documents.
Always cite your sources clearly with:
- Author(s)
- Document title
- Page number

📚 Scientific Literature Search:
When asked for literature on a topic:
- Use combinations like 'process mining' or logic like AND/OR to refine queries.
- Search using both arxiv_tool and google_scholar_tool.
- Return exactly 5 relevant papers (unless otherwise specified).
Each entry must always include:
- Name of research paper
- Author(s) (First author + et al. if applicable)
- Year of publication
- DOI or link
- summary

👤 User Context:
Consider the following user information:
{personal_info}

✅ Tone & Style:
Use a professional and scientific tone.
Ensure clarity, accuracy, and cite evidence.
Be proactive and comprehensive.
"""

# Define Agents
def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt_call_llm)] + list(state["messages"])
    response = llm.invoke(messages)

    return {
        "messages": state["messages"] + [response],
        "last_ai_content": response.content  # Save last AI message
    }

# Create tool node
tool_node = ToolNode(tools)

############# Build graph ################
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    'llm',
    should_continue,
    {True: "tools",
     False: END}
)
graph.add_edge("tools", "llm")
graph.set_entry_point("llm")

memory = MemorySaver()

agent = graph.compile(checkpointer=memory)

def running_agent():
    print("\n=== RAG AGENT ===")

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        config = {"configurable": {"thread_id": "1"}}
        messages = [HumanMessage(content=user_input)]
        print(f"\n👤 USER: {user_input}")

        result = agent.invoke({"messages": messages}, config)

        # First: print all tool calls from AI messages
        print("\n🛠️ Tool Calls:")
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"🔧 Tool: {call['name']}")
                        print(f"📤 Args: {call['args']}")

        print(f"\n🤖 AI: {result['messages'][-1].content}")

if __name__ == "__main__":
    running_agent()