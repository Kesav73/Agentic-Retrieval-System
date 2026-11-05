# Multi Agent
import threading
import os
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langgraph.graph import END, StateGraph, START
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from tools import tool_node
from agents import researcher_node, document_processor_node, synthesizer_node

retriever = None


class AgentState(TypedDict):
    """
    This defines the object that is passed between each node
    in the graph. We will create different nodes for each agent and tool
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


""" Define Edge Logic

We can define some of the edge logic that is needed to decide what to do based on results of the agents
"""


def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return END
    return "continue"


"""
Define the Graph
We can now put it all together and define the graph!
"""

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("DocumentProcessor", document_processor_node)
workflow.add_node("Synthesizer", synthesizer_node)
workflow.add_node("call_tool", tool_node)

# Update conditional edges
workflow.add_conditional_edges(
    "Researcher",
    router,
    {
        "continue": "DocumentProcessor",
        "call_tool": "call_tool",  # Add routing to tool_node
        END: END,
    },
)

workflow.add_conditional_edges(
    "DocumentProcessor",
    router,
    {
        "continue": "Synthesizer",
        "call_tool": "call_tool",  # Add routing to tool_node
        END: END,
    },
)

workflow.add_conditional_edges(
    "Synthesizer",
    router,
    {
        "continue": "Researcher",
        "call_tool": "call_tool",  # Add routing to tool_node
        END: END,
    },
)

# Route back to the original agent after tool execution
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "DocumentProcessor": "DocumentProcessor",
        "Synthesizer": "Synthesizer",
    },
)

workflow.add_edge(START, "Researcher")
graph = workflow.compile()


chat_history = []  # Store the conversation history


def run_workflow(query, data_retriever):
    global chat_history  # Access the global chat history    
    isFileUploaded = False
    if data_retriever:
        isFileUploaded = True
        print("Retriever is initialized")

    # Append user message to chat history
    chat_history.append(
        HumanMessage(
            content=query
            + ("\n(The user has uploaded a file; use user_file_retriever_tool.)" if isFileUploaded else ""),
        )
    )
    try:
        events = graph.stream(
            {
                "messages": chat_history,
            },
            # Maximum number of steps to take in the graph
            {"recursion_limit": 50},
        )

        final_response = ""  # Store the final response

        for s in events:
            print(s)
            if "FINAL ANSWER" in s[list(s.keys())[0]]["messages"][-1].content:
                final_response = s[list(s.keys())[0]]["messages"][-1].content
                # Append agent's response to chat history
                chat_history.append(s[list(s.keys())[0]]["messages"][-1])
                break  # Exit loop when final answer is found

        return final_response
    except Exception as e:
        print(f"error occured: {e}")
        return str(
            "Sorry, I am unable to process your request at the moment. Please try again later."
        )
