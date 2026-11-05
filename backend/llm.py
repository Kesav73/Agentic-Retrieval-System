import os

from dotenv import load_dotenv

load_dotenv(".env")

# Pick the chat model with env vars, e.g. LLM_PROVIDER=groq LLM_MODEL=llama-3.3-70b-versatile
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
MODEL = os.getenv("LLM_MODEL")

if PROVIDER == "google":
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=MODEL or "gemini-2.5-flash", temperature=0)
elif PROVIDER == "groq":
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=MODEL or "llama-3.3-70b-versatile", temperature=0, max_retries=2)
else:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL or "gpt-4o-mini", temperature=0)
