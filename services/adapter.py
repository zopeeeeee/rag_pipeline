# services/adapter.py
import os, time, json
import requests

# Try import of google-genai safely
try:
    from google import genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# If you prefer direct REST rather than SDK, you can set a GEMINI_API_URL, but
# using google-genai SDK is simpler (it reads GEMINI_API_KEY).
GEMINI_API_URL = os.getenv("GEMINI_API_URL", None)

PHI3_LOCAL_URL = os.getenv("PHI3_LOCAL_URL", "http://localhost:11434")

def call_gemini_with_sdk(prompt, model="gemini-2.5-flash", timeout=20):
    if not HAS_GENAI:
        raise RuntimeError("google-genai SDK not installed")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client()  # client picks up GEMINI_API_KEY from env
    resp = client.models.generate_content(model=model, contents=prompt, timeout=timeout)
    return getattr(resp, "text", str(resp))

def call_gemini_rest(prompt, model="gemini-2.5-flash", timeout=20):
    url = GEMINI_API_URL or f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # Extract text safely:
    return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

def call_ollama(prompt, model="phi3:mini", timeout=20):
    # Ollama's REST API: POST /api/generate {"model":"phi3:mini", "prompt":"..."}
    url = f"{PHI3_LOCAL_URL}/api/generate"
    payload = {"model": model, "prompt": prompt}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response") or r.text

def call_mock(prompt):
    # Simple deterministic mock — excellent for testing pipeline
    return "MOCK ANSWER ✅ (pipeline OK — no real LLM configured)."

def call_llm_with_fallback(prompt):
    """
    Tries: Gemini (SDK if available, else REST) -> Ollama local -> MOCK
    returns dict {"answer": str, "provider": "gemini|phi-3|mock", "timestamp": ...}
    """
    # Try Gemini SDK first if available & key is set
    if GEMINI_API_KEY and HAS_GENAI:
        try:
            return {"answer": call_gemini_with_sdk(prompt), "provider": "gemini-sdk", "timestamp": time.time()}
        except Exception as e:
            print("Gemini SDK call failed:", e)

    # Try Gemini REST if key set and GEMINI_API_URL provided or default target
    if GEMINI_API_KEY:
        try:
            return {"answer": call_gemini_rest(prompt), "provider": "gemini-rest", "timestamp": time.time()}
        except Exception as e:
            print("Gemini REST call failed:", e)

    # Try Ollama local
    try:
        return {"answer": call_ollama(prompt), "provider": "phi-3-local", "timestamp": time.time()}
    except Exception as e:
        print("Ollama call failed:", e)

    # Final fallback: mock
    return {"answer": call_mock(prompt), "provider": "mock", "timestamp": time.time()}
