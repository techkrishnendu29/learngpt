import numpy as np
from app.models import nli_model

def softmax(x):
    exp_x = np.exp(
        x - np.max(x)
    )
    return exp_x / exp_x.sum()


def verify_claim(claim, evidences):
    if not evidences:
        return {
            "verdict": "UNCERTAIN",
            "score": 0.0,
            "evidence": None
        }

    best_contradiction = 0
    best_entailment = 0
    best_contradiction_evidence = None
    best_support_evidence = None

    try:
        pairs = [
            (evidence[:512], claim)
            for evidence in evidences
        ]
        logits = nli_model.predict(pairs)
        for evidence, logit in zip(evidences, logits):
            probs = softmax(
                np.array(logit)
            )
            # Label order:
            # 0 = contradiction
            # 1 = neutral
            # 2 = entailment
            contradiction = probs[0]
            neutral = probs[1]
            entailment = probs[2]
            if contradiction > best_contradiction:
                best_contradiction = contradiction
                best_contradiction_evidence = evidence
            if entailment > best_entailment:
                best_entailment = entailment
                best_support_evidence = evidence
        if best_entailment >= 0.75:
            return {
                "verdict": "SUPPORTED",
                "score": float(best_entailment),
                "evidence": best_support_evidence
            }
        elif best_contradiction >= 0.75:
            return {
                "verdict": "HALLUCINATED",
                "score": float(best_contradiction),
                "evidence": best_contradiction_evidence
            }
        else:
            return {
                "verdict": "UNCERTAIN",
                "score": float(
                    max(
                        best_entailment,
                        best_contradiction
                    )
                ),
                "evidence": None
            }
    except Exception as e:
        print("Verification Error:", e)
        return {
            "verdict": "UNCERTAIN",
            "score": 0.0,
            "evidence": None
        }