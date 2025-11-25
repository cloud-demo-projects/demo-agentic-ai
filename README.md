# Layers
- Azure Python FunctionApp -> API
- LangChain -> Agent
- LLM -> Ollama
- Search Engine -> Duckduckgo

<img width="3964" height="1648" alt="image" src="https://github.com/user-attachments/assets/c9a8be08-0f8a-4182-85db-b75728449e72" />

## Local Installation Steps
1️⃣ Install or Update Ollama (on your local dev machine)
For Linux/Mac:
curl -fsSL https://ollama.com/install.sh | sh

For Windows:
Download from https://ollama.com/download
Run installer and restart your terminal.
Confirm it’s working:
ollama --version
should show >= 0.1.46

2️⃣ Pull a Supported Model
For example:
ollama pull llama3
or smaller model:
ollama pull phi3

✅ Tip:
llama3 gives best quality for reasoning;
phi3 is smallest (good for local tests).

3️⃣ Start Ollama Server (auto-starts in background)
ollama server
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

