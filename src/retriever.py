"""
retriever.py
------------
Retrieval component of the RAG pipeline. Given a new incoming email, finds
the k most similar historical (email, reply) pairs from the training split.

Why TF-IDF instead of a neural embedding model: this environment has no
network access to download embedding model weights, and the point of this
exercise is the *system design*, not the retrieval model. TF-IDF + cosine
similarity is a legitimate, fully offline, dependency-light retriever that
works well in a narrow domain like support email (lots of shared vocabulary:
"order", "refund", "subscription", etc.). The interface below is written so
swapping in a real embedding model (e.g. `sentence-transformers` or the
Anthropic/OpenAI embeddings API) later is a one-class change -- see
`EmbeddingRetriever` stub at the bottom.
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, corpus_path):
        self.rows = [json.loads(l) for l in open(corpus_path)]
        texts = [r["subject"] + "\n" + r["incoming_email"] for r in self.rows]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query_subject, query_body, k=3, exclude_id=None):
        q = self.vectorizer.transform([query_subject + "\n" + query_body])
        sims = cosine_similarity(q, self.matrix)[0]
        order = sims.argsort()[::-1]
        out = []
        for idx in order:
            row = self.rows[idx]
            if exclude_id is not None and row["id"] == exclude_id:
                continue
            out.append((row, float(sims[idx])))
            if len(out) >= k:
                break
        return out


class EmbeddingRetriever:
    """
    Drop-in replacement for TfidfRetriever using real embeddings (e.g. via an
    API). Not used by default (no network in this environment) but included
    so the trade-off is explicit and swapping is trivial:

        def embed(texts) -> np.ndarray: call your embeddings endpoint
        then cosine-similarity same as above.

    Trade-off vs TF-IDF: better at paraphrase/semantic matches ("my package
    never came" ~ "where is my order"), at the cost of needing network + an
    embeddings API/model and a bit more latency and cost per call.
    """
    pass
