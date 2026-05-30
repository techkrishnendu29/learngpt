from fastapi import FastAPI
from pydantic import BaseModel

from app.llm import generate_llm_response
from app.claims import extract_claims
from app.retrieval import retrieve_best_evidence
from app.verification import verify_claim
from app.correction import generate_correct_answer

app = FastAPI(
    title="VeriGPT API",
    description="Educational AI Assistant with Hallucination Detection",
    version="1.0"
)

class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():

    return {
        "message": "VeriGPT API Running Successfully"
    }

@app.post("/ask")
def ask_verigpt(data: QueryRequest):
    query = data.query
    ai_response = generate_llm_response(query)
    if not ai_response:
        return {
            "success": False,
            "message": "Failed to generate AI response"
        }
    claims = extract_claims(ai_response)
    if not claims:
        return {
            "success": True,
            "query": query,
            "llm_response": ai_response,
            "claims": [],
            "hallucination_score": 0,
            "status": "UNCERTAIN"
        }

    all_results = []
    for claim in claims:
        evidences = retrieve_best_evidence(claim)
        result = verify_claim(
            claim,
            evidences
        )
        all_results.append({
            "claim": claim,
            "verdict": result["verdict"],
            "confidence": round(
                result["score"] * 100,
                2
            ),
            "evidence": result["evidence"]
        })
    hallucinated_claims = [
        r for r in all_results
        if r["verdict"] == "HALLUCINATED"
    ]
    hallucination_score = (
        len(hallucinated_claims)
        / len(all_results)
    ) * 100

    corrected_answer = None
    status = "VERIFIED"
    if hallucination_score > 0:
        status = "HALLUCINATION DETECTED"
        best_evidence = hallucinated_claims[0]["evidence"]
        corrected_answer = generate_correct_answer(
            query,
            best_evidence
        )

    return {
        "success": True,
        "query": query,
        "llm_response": ai_response,
        "claims": all_results,
        "hallucination_score": round(
            hallucination_score,
            2
        ),
        "status": status,
        "corrected_answer": corrected_answer
    }