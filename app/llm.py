from app.config import client
def generate_llm_response(query):
    try:
        prompt = f"""
        You are a factual educational assistant.
        Answer the following query clearly and accurately.
        Query:
        {query}
        Keep the answer concise, educational,
        and fact-based.
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        if response and hasattr(response, "text"):
            return response.text
        return None
    except Exception as e:
        print("Gemini Error:", e)
        return None