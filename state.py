"""
Agent state schema for the simulation runner.

This is the shared state that flows through the LangGraph graph.
Each node reads from and writes to this state.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State passed between nodes in the simulation agent graph.

    messages:  Chat history (user + assistant + tool messages).
               `add_messages` is a LangGraph reducer that appends
               new messages instead of overwriting.
    """
    messages: Annotated[list, add_messages]
