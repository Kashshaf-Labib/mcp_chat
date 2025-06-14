import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
import os


async def run_memory_chat():
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    config_file = "browser_mcp.json"

    print("initializing client")
    try:
        client = MCPClient.from_config_file(config_file)
        
        # Ensure all MCP servers are running
        await client.ensure_servers_running()
        
        llm = ChatGroq(model="qwen-qwq-32b")
        agent = MCPAgent(llm=llm, client=client, max_steps=30, memory_enabled=True)

        print("\n====== Interactive MCP Chat ======")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'clear' to clear the memory")
        print("=================================")

        while True:
            user_input = input("\nYou: ")

            if user_input.lower() in ["quit", "exit"]:
                print("Ending Conversation...")
                break

            if user_input.lower() == "clear":
                agent.clear_memory()
                print("Memory Cleared")
                continue

            print("\nAssistant: ", end="", flush=True)

            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"\nError: {e}")
                # Attempt to reconnect if connection is lost
                if "Connection closed" in str(e):
                    print("Attempting to reconnect...")
                    await client.ensure_servers_running()
                    continue

    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        print("\nEnding Conversation...")
        if client and client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    print("Starting Conversation...")
    asyncio.run(run_memory_chat())
