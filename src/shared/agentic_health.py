import os
"""LangChain helpers for generating nutrition-focused plans."""

import logging
from langchain_community.chat_models import (
    AzureChatOpenAI,
    ChatOpenAI,
    ChatOllama
)
import os
from typing import List, Tuple
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_community.chat_models import AzureChatOpenAI, ChatOllama, ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory


# -------------------------------------------------------------------
# Feature flags / environment
# -------------------------------------------------------------------

ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() not in {
    "0",
    "false",
    "no",
}


# -------------------------------------------------------------------
# LLM selection
# -------------------------------------------------------------------

def _current_model_signature() -> Tuple[str, str, str, str, str, str]:
    """Capture relevant configuration inputs so caches react to env changes."""

    return (
        os.getenv("AZURE_OPENAI_API_KEY", ""),
        os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        os.getenv("OPENAI_API_KEY", ""),
        os.getenv("OPENAI_MODEL", ""),
        os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
    )


def _build_llm(signature: Tuple[str, str, str, str, str, str]):
    """Choose the best available model: Azure → OpenAI → Local Ollama."""

    azure_key, azure_endpoint, azure_deployment, openai_key, openai_model, ollama_model = signature

    try:
        # Prefer Azure
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
            azure_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
            logging.info("Using Azure OpenAI chat deployment %s", azure_deployment)

            if not (azure_deployment and azure_endpoint and azure_key and api_version):
                missing = [n for n,v in {
                    "AZURE_OPENAI_DEPLOYMENT": azure_deployment,
                    "AZURE_OPENAI_ENDPOINT": azure_endpoint,
                    "AZURE_OPENAI_API_KEY": azure_key,
                    "AZURE_OPENAI_API_VERSION": api_version
                }.items() if not v]
                logging.warning("Azure OpenAI env incomplete (%s); falling back.", ", ".join(missing))
            else:
                return AzureChatOpenAI(
                    azure_deployment=azure_deployment,
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    api_version=api_version,
                    temperature=0.4
                )

        # Fallback to OpenAI
        elif os.getenv("OPENAI_API_KEY"):
            logging.info("Using OpenAI API...")
        if openai_key:
            logging.info("Using OpenAI chat model %s", openai_model)
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", ""),
                temperature=0.4
            )

        # Final fallback to local Ollama
        logging.warning("No cloud API key found — using local Ollama model %s.", ollama_model)
        return ChatOllama(model=ollama_model)

    except Exception as exc:
        logging.error("Model selection failed: %s", exc)
        return ChatOllama(model=ollama_model)

# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------
def get_llm():
    """
    Public helper for callers that only need an LLM instance.
    Cached so we don't re-create LLM clients on every call.
    """
    signature = _current_model_signature()
    return _build_llm(signature)


# -------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------
def build_tools() -> List[Tool]:
    """Create the list of tools used by the agent."""
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

    return tools


# -------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------
def _get_agent(tools: List[Tool], llm, memory):
    """Construct an agent that works reliably with memory and tools."""

    if not tools:
        return None

    try:
        return initialize_agent(
            tools=tools,
            llm=llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors="Check your output and try again.",
            max_iterations=3,
            max_execution_time=60,
            early_stopping_method="generate",
        )
    except Exception as exc:
        logging.exception("Agent initialization failed: %s", exc)
        return None


def generate_meal_plan(goal: str, purpose: str):
    """Generate an anti-inflammatory, low-glycemic meal plan."""

    # 1. Build LLM
    llm = get_llm()

    # 2. Build tools
    tools = build_tools()

    # 3. Build memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    # 4. Build agent
    agent = _get_agent(tools, llm, memory)
    if agent is None:
        raise ValueError("No tools available for the agent to use.")

    # 5. Create Agent Prompt with clear output target
    prompt = (
        "You are a nutrition coach. Create a concise, actionable "
        "anti-inflammatory, low-glycemic meal plan. "
        "If needed, use the Web Search tool for evidence. "
        "Return the final plan under 'Final Answer:' as bullet points.\n\n"
        f"Goal: {goal}\nPurpose: {purpose}"
    )

    try:
        # 6. Run agent using invoke for LangChain 0.2
        if agent is not None:
            agent_result = agent.invoke({"input": prompt})
            output_text = (
                agent_result["output"]
                if isinstance(agent_result, dict) and "output" in agent_result
                else agent_result
            )
            return {
                "goal": goal,
                "purpose": purpose,
                "meal_plan": output_text,
                "model": type(llm).__name__
            }

        # Fallback: direct LLM without tools
        logging.warning("Agent unavailable; falling back to direct LLM predict.")
        direct = llm.predict(prompt)
        return {
            "goal": goal,
            "purpose": purpose,
            "meal_plan": direct,
            "model": type(llm).__name__
        }

    except Exception as exc:
        logging.exception("Agent execution failed; falling back to direct LLM.")
        try:
            direct = llm.predict(prompt)
            return {
                "goal": goal,
                "purpose": purpose,
                "meal_plan": direct,
                "model": type(llm).__name__
            }
        except Exception as exc2:
            return {
                "goal": goal,
                "purpose": purpose,
                "error": f"Agent error: {exc}; LLM fallback error: {exc2}",
                "model": type(llm).__name__
            }
