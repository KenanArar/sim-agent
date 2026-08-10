"""
LTspice Simulation Agent — Interactive CLI.

Routes requests to specialized subagents:
  Level 1 (simulation): run/interpret existing circuits
  Level 2 (generator):  design new circuits from specs

Usage:
    python main.py
"""

from langchain_core.messages import HumanMessage
from agents.root_agent import build_graph


def main():
    print("=" * 60)
    print("  LTspice Simulation Agent (Levels 1 + 2)")
    print("=" * 60)
    print()
    print("Commands:")
    print("  Type a message to interact with the agent")
    print("  'quit' or 'exit' to stop")
    print("  'clear' to reset conversation")
    print()

    graph = build_graph()
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            messages = []
            print("Conversation cleared.\n")
            continue

        messages.append(HumanMessage(content=user_input))

        # Stream the agent's response
        print("\nAgent: ", end="", flush=True)

        result = graph.invoke({"messages": messages})

        # Get the final assistant message
        assistant_msg = result["messages"][-1]
        print(assistant_msg.content)
        print()

        # Update message history
        messages = result["messages"]


if __name__ == "__main__":
    main()
