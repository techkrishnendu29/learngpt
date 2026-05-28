import requests
import numpy as np
import faiss

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor

from app.models import embedding_model

def search_web(query, max_results=5):
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                max_results=max_results
            )
            for r in results:
                href = r.get("href")
                if href:
                    urls.append(href)
    except Exception as e:
        print("Search Error:", e)
    return list(set(urls))

def fetch_page_text(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(
            url,
            headers=headers,
            timeout=8
        )
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(
            response.text,
            "lxml"
        )
        paragraphs = soup.find_all("p")
        text = " ".join(
            p.get_text(" ", strip=True)
            for p in paragraphs[:25]
        )
        if len(text) < 300:
            return None
        return text[:4000]
    except Exception as e:
        print("Fetch Error:", e)
        return None

def collect_web_evidence(claim):
    urls = search_web(claim)
    if not urls:
        return []
    docs = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(
            fetch_page_text,
            urls
        )
        for content in results:
            if content:
                docs.append(content)
    return docs

def rank_evidence(claim, docs, top_k=3):
    if not docs:
        return []
    try:
        doc_embeddings = embedding_model.encode(
            docs,
            normalize_embeddings=True
        ).astype("float32")
        claim_embedding = embedding_model.encode(
            [claim],
            normalize_embeddings=True
        ).astype("float32")
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(doc_embeddings)
        scores, indices = index.search(
            claim_embedding,
            min(top_k, len(docs))
        )
        ranked_docs = []
        for idx in indices[0]:
            ranked_docs.append(docs[idx])
        return ranked_docs
    except Exception as e:
        print("FAISS Error:", e)
        return []

def retrieve_best_evidence(claim):
    docs = collect_web_evidence(claim)
    if not docs:
        return []
    ranked_docs = rank_evidence(
        claim,
        docs
    )
    return ranked_docs