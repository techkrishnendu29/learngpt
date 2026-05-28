import spacy
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)
print("Loading spaCy model...")

nlp = spacy.load(
    "en_core_web_sm"
)
print("Loading embedding model...")
embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)
print("Loading NLI verification model...")
nli_model = CrossEncoder(
    "cross-encoder/nli-MiniLM2-L6-H768"
)
print("All models loaded successfully.")