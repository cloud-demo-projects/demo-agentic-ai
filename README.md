## Layers
- API -> Azure Python FunctionApp
- Agent -> LangChain
- LLM -> Ollama: llama3.2:1b ; Azure Open AI: GPT-4.1
- Tools -> Search Engine: Duckduckgo

## Visualization 1
<img width="3964" height="1648" alt="image" src="https://github.com/user-attachments/assets/c9a8be08-0f8a-4182-85db-b75728449e72" />

## Visualization 2
<img width="3964" height="1648" alt="image" src="https://learn.microsoft.com/en-us/agent-framework/media/agent.svg" />


## Local Ollama Installation Steps
1️⃣ Install or Update Ollama (on your local dev machine)
Confirm it’s working: ollama --version

2️⃣ Pull a Supported Model
For example: ollama pull llama3

3️⃣ Start Ollama Server (auto-starts in background)
ollama serve
ollama runs at http://localhost:11434 by default.

4️⃣ Python Package Setup
Make sure you have these versions (compatible with LangChain 0.2.x):
pip install "langchain-community==0.2.12" "langchain==0.2.14"
You do not need any separate ollama Python package — LangChain directly integrates through the local API.

5️⃣ Example Test Script
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="llama3", temperature=0.4)
response = llm.invoke("Explain how flaxseeds help reduce inflammation.")
print(response)
✅ Expected: A natural-language answer directly from your local model

