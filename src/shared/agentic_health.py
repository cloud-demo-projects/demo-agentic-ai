import os
import logging
from langchain_community.chat_models import (
    AzureChatOpenAI,
    ChatOpenAI,
    ChatOllama
)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory


def get_llm():
    """Choose the best available model: Azure → OpenAI → Local Ollama."""
    try:
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            logging.info("Using Azure OpenAI...")
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-05-01-preview",
                temperature=0.4
            )

        elif os.getenv("OPENAI_API_KEY"):
            logging.info("Using OpenAI API...")
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.4
            )

        else:
            logging.warning("No cloud API key found — using local Ollama model.")
            return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"))

    except Exception as e:
        logging.error(f"Model selection failed: {e}")
        # Final fallback to local Ollama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"))


def generate_meal_plan(goal: str):
    """Generate an anti-inflammatory, low-glycemic meal plan."""
    llm = get_llm()
    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="Web Search",
            func=search.run,
            description="Search the web for nutritional and metabolic evidence."
        )
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent = initialize_agent(
        tools, llm,
        agent_type="conversational-react-description",
        memory=memory,
        verbose=False
    )

    try:
        result = agent.run(f"Create an anti-inflammatory, low-glycemic meal plan for: {goal}")
        return {"goal": goal, "meal_plan": result, "model": type(llm).__name__}
    except Exception as e:
        logging.error(f"Agent execution failed: {e}")
        return {"goal": goal, "error": str(e), "model": type(llm).__name__}
