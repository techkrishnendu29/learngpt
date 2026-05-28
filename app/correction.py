from app.config import client

def generate_correct_answer(query, evidence):
    if not evidence:
        return None
    try:
        prompt = f"""
        The previous AI-generated response may contain
        factual inaccuracies.
        User Question:
        {query}
        Verified Evidence:
        {evidence[:2000]}
        Instructions:
        1. Use ONLY the verified evidence.
        2. Do not add external information.
        3. Keep the answer concise and factual.
        4. If evidence is insufficient, clearly say so.
        Generate the corrected factual answer.
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        if response and hasattr(response, "text"):
            return response.text
        return None
    except Exception as e:
        print("Correction Error:", e)
        return None