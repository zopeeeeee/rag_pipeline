# test_gemini.py
from google import genai

# The client will read GEMINI_API_KEY from env automatically
client = genai.Client()

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain RAG pipelines in two sentences."
)
print("=== RESPONSE TEXT ===")
print(resp.text)
