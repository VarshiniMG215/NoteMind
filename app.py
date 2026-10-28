"""Flask REST API + CLI for NoteMind.

Run API:  python app.py
Run CLI:  python app.py --cli
"""
import sys

from flask import Flask, jsonify, request

from rag import NoteMind

app = Flask(__name__)
bot = NoteMind("knowledge")


@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Send JSON like {\"question\": \"...\"}"}), 400
    return jsonify(bot.ask(question))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "chunks_indexed": len(bot.chunks)})


if __name__ == "__main__":
    if "--cli" in sys.argv:
        while True:
            q = input("Ask your notes (or 'quit'): ").strip()
            if q.lower() == "quit":
                break
            r = bot.ask(q)
            print(r["answer"], "\nSources:", r["sources"], "\n")
    else:
        app.run(debug=True)
