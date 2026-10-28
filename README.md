# NoteMind – RAG Study Assistant

Ask questions about your own study notes. NoteMind retrieves the most relevant
note chunks (TF-IDF vector search) and sends only those to an LLM (Claude API),
which answers from that context and cites the source file.

## Concepts demonstrated
- Retrieval-Augmented Generation (chunking, vectorization, cosine-similarity top-k)
- LLM API integration with a system prompt that grounds answers to reduce hallucination
- Context-window awareness (only top-k chunks are sent, not all notes)
- Flask REST API (`POST /ask`) + CLI

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key      # Windows: set ANTHROPIC_API_KEY=your_key
python app.py --cli                    # or: python app.py
```
API example:
```bash
curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" \
     -d '{"question": "What are the four conditions for deadlock?"}'
```
Add your own `.txt` notes to `knowledge/`.

## Limitations / next steps
- TF-IDF matches keywords, not meaning. Next: swap in embeddings + a vector DB (FAISS/Chroma).
- Add tool/function calling and conversation memory.
