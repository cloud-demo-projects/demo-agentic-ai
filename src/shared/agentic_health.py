import os
"""LangChain helpers for generating nutrition-focused plans."""

import logging
from langchain_community.chat_models import (
    AzureChatOpenAI,
    ChatOpenAI,
    ChatOllama
)
import os
from functools import lru_cache
from typing import List, Tuple

from langchain.agents import AgentType, Tool, initialize_agent
from langchain_community.chat_models import AzureChatOpenAI, ChatOllama, ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

# Feature flag to disable web search when low latency or offline execution is
# desired. Any truthy value keeps the search tool enabled.
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() not in {"0", "false", "no"}

def _current_model_signature() -> Tuple[str, str, str, str, str, str]:
    """Capture relevant configuration inputs so caches react to env changes."""

    return (
        os.getenv("AZURE_OPENAI_API_KEY", ""),
        os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        os.getenv("OPENAI_API_KEY", ""),
        os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        os.getenv("OLLAMA_MODEL", "llama3"),
    )


def _build_llm(signature: Tuple[str, str, str, str, str, str]):
    """Choose the best available model: Azure → OpenAI → Local Ollama."""

    azure_key, azure_endpoint, azure_deployment, openai_key, openai_model, ollama_model = signature

    try:
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            logging.info("Using Azure OpenAI...")
        if azure_key and azure_endpoint:
            logging.info("Using Azure OpenAI chat deployment %s", azure_deployment)
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-05-01-preview",
                temperature=0.4
            )
        elif os.getenv("OPENAI_API_KEY"):
            logging.info("Using OpenAI API...")
        if openai_key:
            logging.info("Using OpenAI chat model %s", openai_model)
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.4
            )

        logging.warning("No cloud API key found — using local Ollama model %s.", ollama_model)
        return ChatOllama(model=ollama_model)

    except Exception as exc:
        logging.error("Model selection failed: %s", exc)
        return ChatOllama(model=ollama_model)


    except Exception as e:
        logging.error(f"Model selection failed: {e}")
        # Final fallback to local Ollama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"))


def get_llm():
    """Public helper for callers that only need an LLM instance."""

    return _build_llm(_current_model_signature())


def generate_meal_plan(goal: str, purpose: str):
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
    signature = _current_model_signature()
    agent, llm = _get_agent(signature)
    prompt = f"{purpose}: {goal}"

    try:
        result = agent.run(f"{purpose}: {goal}")
        if agent:
            result = agent.run(prompt)
        else:
            result = llm.predict(prompt)
        return {"goal": goal, "purpose": purpose, "meal_plan": result, "model": type(llm).__name__}
    except Exception as e:
        logging.error(f"Agent execution failed: {e}")
        return {"goal": goal, "purpose": purpose, "error": str(e), "model": type(llm).__name__}
    except Exception as exc:
        logging.error("Agent execution failed: %s", exc)
        return {"goal": goal, "purpose": purpose, "error": str(exc), "model": type(llm).__name__}


@lru_cache(maxsize=1)
def _get_agent(signature: Tuple[str, str, str, str, str, str]):
    """Construct and cache the agent pipeline keyed by configuration inputs."""

    llm = _build_llm(signature)

    tools: List[Tool] = []
    if ENABLE_WEB_SEARCH:
        search = DuckDuckGoSearchRun()
        tools.append(
            Tool(
                name="Web Search",
                func=search.run,
                description="Search the web for nutritional and metabolic evidence.",
            )
        )

    if tools:
        agent = initialize_agent(
            tools,
            llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )
    else:
        agent = None

    return agent, llm