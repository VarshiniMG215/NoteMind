"""NoteMind core: a tiny Retrieval-Augmented Generation (RAG) pipeline.

Steps: load notes -> chunk -> vectorize (TF-IDF) -> retrieve top-k by cosine
similarity -> send only those chunks to an LLM -> answer with sources.
"""
import glob
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

MODEL = "claude-sonnet-5-5"
CHUNK_WORDS = 80
OVERLAP = 20


def load_chunks(folder="knowledge"):
    """Read every .txt file and split into overlapping word chunks."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        step = CHUNK_WORDS - OVERLAP
        for i in range(0, max(len(words), 1), step):
            text = " ".join(words[i:i + CHUNK_WORDS])
            if text:
                chunks.append({"source": os.path.basename(path), "text": text})
    return chunks


class NoteMind:
    def __init__(self, folder="knowledge"):
        self.chunks = load_chunks(folder)
        if not self.chunks:
            raise ValueError(f"No .txt notes found in '{folder}/'")
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in self.chunks)
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def retrieve(self, question, k=3):
        """Return the k most similar chunks (vector search over TF-IDF)."""
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [dict(self.chunks[i], score=float(scores[i])) for i in top if scores[i] > 0]

    def ask(self, question):
        hits = self.retrieve(question)
        if not hits:
            return {"answer": "I couldn't find anything relevant in your notes.", "sources": []}
        context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
        # Grounding instruction reduces hallucination: answer ONLY from context.
        system = (
            "You are a study assistant. Answer using ONLY the context provided. "
            "If the context does not contain the answer, say you don't know. "
            "Keep answers short and mention the source file name."
        )
        msg = self.client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=system,
            messages=[{"role": "user",
                       "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
        return {"answer": msg.content[0].text,
                "sources": sorted({h["source"] for h in hits})}
