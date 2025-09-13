import os, time, requests
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Optional: only import gemini if API key exists
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
PHI3_LOCAL_URL = os.getenv("PHI3_LOCAL_URL", "http://localhost:11434")

try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        HAS_GEMINI = True
    else:
        HAS_GEMINI = False
except Exception as e:
    print("Gemini import/config failed:", e)
    HAS_GEMINI = False


def call_gemini(prompt, model=GEMINI_MODEL, timeout=20):
    if not HAS_GEMINI:
        raise RuntimeError("Gemini API key not set or SDK missing")
    try:
        llm_model = genai.GenerativeModel(model)
        resp = llm_model.generate_content(prompt, request_options={"timeout": timeout})
        return resp.text
    except Exception as e:
        print("Gemini call failed:", e)
        raise


def call_ollama(prompt, model="phi3", timeout=20):
    url = f"{PHI3_LOCAL_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response") or str(data)
    except Exception as e:
        print("Ollama call failed:", e)
        raise


def call_mock(prompt):
    print("Using MOCK response ✅")
    return "MOCK ANSWER ✅ (pipeline OK — no real LLM configured)."


def call_llm_with_fallback(prompt):
    """
    Try Gemini → Ollama → Mock
    Returns dict {"answer": str, "provider": "gemini|phi-3|mock", "timestamp": ...}
    """
    # Try Gemini
    if HAS_GEMINI:
        try:
            answer = call_gemini(prompt)
            print("Answered by Gemini ✅, Gemini Raw Response:", repr(answer))
            return {"answer": answer, "provider": "gemini", "timestamp": time.time()}
        except Exception:
            pass

    # Try Ollama
    try:
        answer = call_ollama(prompt)
        print("Answered by Ollama ✅")
        return {"answer": answer, "provider": "phi-3-local", "timestamp": time.time()}
    except Exception:
        pass

    # Fallback
    answer = call_mock(prompt)
    return {"answer": answer, "provider": "mock", "timestamp": time.time()}
