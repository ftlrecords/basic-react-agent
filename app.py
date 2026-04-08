from agents.react_agent import build_agent
from langchain_core.messages import HumanMessage

def main():
    agent = build_agent()

    print("🤖 LangGraph Multi-Tool Agent (type 'exit' to quit)\n")

    messages = []

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        messages.append(HumanMessage(content=user_input))

        response = agent.invoke({
            "messages": messages
        })

        # update full conversation
        messages = response["messages"]

        print("\n--- Agent Response ---")
        for msg in messages[-3:]:   # show last few messages
            print(msg.content)
        print("----------------------\n")


if __name__ == "__main__":
    main()