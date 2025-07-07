import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from agent.state import OverallState
from agent.configuration import Configuration
from agent.nodes.audit_coordinator import audit_coordinator_agent

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")


def should_continue(state: OverallState) -> str:
    """Router function to determine if audit should continue or complete"""
    print(f"=== DEBUG should_continue: Checking if should continue ===")
    
    # Always complete after processing a message to avoid infinite loops
    # The agent should only run once per user input
    messages = state.get("messages", [])
    print(f"DEBUG: Total messages in state: {len(messages)}")
    
    # If we have processed the user input, we should stop
    # The agent will be called again when the user sends a new message
    if len(messages) >= 2:  # At least one user message and one assistant response
        last_message = messages[-1] if messages else None
        if last_message and hasattr(last_message, '__class__') and 'AIMessage' in str(type(last_message)):
            print(f"DEBUG: Last message is AI message, completing conversation turn")
            return "complete"
    
    print(f"DEBUG: Continuing conversation")
    return "continue"


# Create our Agent Graph - Simple single coordinator pattern
builder = StateGraph(OverallState, config_schema=Configuration)

# Add single audit coordinator node
builder.add_node("audit_coordinator", audit_coordinator_agent)

# Set entrypoint
builder.add_edge(START, "audit_coordinator")

# Add conditional edges for continuation
builder.add_conditional_edges("audit_coordinator", should_continue, {
    "continue": "audit_coordinator",
    "complete": END
})

graph = builder.compile(name="audit-assistant-agent")
