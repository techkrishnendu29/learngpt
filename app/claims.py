from app.models import nlp
def extract_claims(text):
    doc = nlp(text)
    claims = []
    for sent in doc.sents:
        sentence = sent.text.strip()
        if len(sentence.split()) < 5:
            continue
        if any(
            tok.pos_ in ["VERB", "AUX"]
            for tok in sent
        ):
            claims.append(sentence)
    return claims[:6]