import imaplib
import email
from email.header import decode_header
from dotenv import load_dotenv
import os
from datetime import datetime

from typing import Annotated, TypedDict, List, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv()

# CONFIGURATION
IMAP_SERVER = 'imap.web.de'
EMAIL_ACCOUNT = os.getenv('EMAIL_ACCOUNT')
PASSWORD = os.getenv('EMAIL_PASSWORD')

today = datetime.today().strftime("%d-%b-%Y")

llm = ChatOpenAI(model= "o4-mini")

# CONNECT TO SERVER
def connect(imap_server: str, email_account: str, password: str) -> imaplib.IMAP4_SSL:
    """
    Function, that connects to the mail server
    """
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(email_account, password)
    return mail

# DECODE HEADERS
def decode_mime_words(header_val):
    "Function, for decoding the headers of the email"
    decoded = decode_header(header_val)
    return ''.join([
        str(t[0], t[1] or 'utf-8') if isinstance(t[0], bytes) else str(t[0])
        for t in decoded
    ])

############### Tools #############

@tool
def get_emails(date: str) -> str:
    """
    function, that takes a date for a specific day and gives back the emails that were recived at this date.
    The date format is "%d-%b-%Y".
    """
    mail = connect(IMAP_SERVER, EMAIL_ACCOUNT, PASSWORD)
    mail.select('INBOX')
    status, messages = mail.search(None, 'ON', date)
    email_ids = messages[0].split()

    full_emails = []

    for num in email_ids[::-1]:
        status, data = mail.fetch(num, '(RFC822)')
        raw_email = data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject = decode_mime_words(msg.get("Subject", ""))
        from_ = decode_mime_words(msg.get("From", ""))
        date_ = msg.get("Date", "")

        # Extract body
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain" and part.get_content_disposition() is None:
                    body = part.get_payload(decode=True).decode(errors='ignore')
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors='ignore')

        # Combine all parts into a single string
        full_email = (
            f"From: {from_}\n"
            f"Subject: {subject}\n"
            f"Date: {date_}\n"
            f"{'-'*50}\n"
            f"{body.strip()}\n"
        )

        full_emails.append(full_email)

    return ' '.join(full_emails)

tools = [get_emails]

llm_with_tools = llm.bind_tools(tools)

########################## Building Agents ###########################

system_message_retriever = f"""
Du bist ein hilfreicher E-Mail-Assistent. Nimm die Anfrage des Nutzers entgegen und führe so viele Tool-Aufrufe wie nötig aus, jedoch nicht mehr als 5.

Hauptziel:

    E-Mails zusammenfassen:
    Wenn der Nutzer darum bittet, E-Mails ab einem bestimmten Datum zusammenzufassen, dann benutze das Tool „get_emails“
    und gib das Datum im Format „%d-%b-%Y“ ein, z. B. „28-May-2025“.
    Falls nach den Emails von heute gefragt wird, dann nimm das datum {today}
"""

system_message_summerizer = f"""
Du bist ein hilfreicher E-Mail-Assistent.

Hauptziel:

    E-Mails zusammenfassen:
    Gib immer die Anzahl an emails an, welche reingekommen sind. z.B. "Es waren 7 Emails in deinem Postfach"
    WICHTIG: Gib niemals die original Emails zurück, sondern immer nur die Zusammenfassung.

    Die E-Mails sind folgendermaßen formatiert:
    From: Wer hat die E-Mail geschrieben
    Subject: Hauptthema der E-Mail
    Date: Datum und Uhrzeit, wann die E-Mail empfangen wurde
    --------------------------------------------------------
    Hauptnachricht der E-Mail
    
    Deine Hauptaufgabe ist es, die E-Mails zusammenzufassen.
    Das bedeutet: Die Hauptaussage jeder einzelnen E-Mail soll so zusammengefasst werden, dass der Kerngedanke erhalten bleibt.

    Für jede E-Mail soll eine eigene Zusammenfassung und ein eigener Aufzählungspunkt erstellt werden.
    """

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    date: str

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

def mail_retriever_agent(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_message_retriever)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}

def mail_summerizer(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_message_summerizer)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}

#################### Build Graph ###########################
graph = StateGraph(AgentState)

graph.add_node("mail_retrierver", mail_retriever_agent)
graph.add_node("tools", ToolNode(tools))
graph.add_node("mail_summerizer", mail_summerizer)

graph.set_entry_point("mail_retrierver")

graph.add_conditional_edges(
    "mail_retrierver",
    should_continue,
    {True: "tools", False: "mail_summerizer"}
)

graph.add_edge("tools", "mail_summerizer")  # ✅ Nur von Tools → Summarizer
graph.add_edge("mail_summerizer", END)


memory = MemorySaver()
agent = graph.compile(checkpointer=memory)

def running_agent():
    print("\n=== EMAIL ASSISTENT ===")

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