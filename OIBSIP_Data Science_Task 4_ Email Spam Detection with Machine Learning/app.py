
import json
import math
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from preprocess import tokenize

BASE = Path(__file__).resolve().parent
PUBLIC = BASE / "public"

with open(BASE / "model" / "model.json") as f:
    MODEL = json.load(f)

TERMS = MODEL["terms"]      # word -> [weight, idf]
BIAS = MODEL["bias"]
MAX_LENGTH = 5000           # characters; plenty for any message

app = Flask(__name__)


def strength(score):
   
    size = abs(score)
    if size < 0.35:
        return "borderline"
    if size < 0.9:
        return "fairly confident"
    return "very confident"


def check_message(text):
    pairs = tokenize(text)

    # Count each known word (by its stem) and remember how it was originally written
    counts, shown_as = {}, {}
    for original, stem in pairs:
        if stem in TERMS:
            counts[stem] = counts.get(stem, 0) + 1
            shown_as.setdefault(stem, original)

    # TF-IDF value of each word, scaled so the whole message has length 1
    tfidf = {s: c * TERMS[s][1] for s, c in counts.items()}
    length = math.sqrt(sum(v * v for v in tfidf.values())) or 1.0

    # How much each word pushed the score towards spam (+) or ham (-)
    pushes = {s: TERMS[s][0] * v / length for s, v in tfidf.items()}
    score = BIAS + sum(pushes.values())

    ordered = sorted(pushes.items(), key=lambda item: item[1])
    spam_clues = [shown_as[s] for s, p in reversed(ordered) if p > 0.02][:6]
    ham_clues = [shown_as[s] for s, p in ordered if p < -0.02][:6]

    return {
        "label": "spam" if score > 0 else "ham",
        "score": round(score, 3),
        "confidence": strength(score),
        "spam_clues": spam_clues,
        "ham_clues": ham_clues,
        "known_words": len(counts),
    }


@app.get("/")
def home():
    # On Vercel the files in /public are served automatically; this is for local use.
    return send_from_directory(PUBLIC, "index.html")


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("message", "")).strip()
    if not text:
        return jsonify(error="Please type or paste a message first."), 400
    if len(text) > MAX_LENGTH:
        return jsonify(error=f"That message is too long (the limit is {MAX_LENGTH:,} characters)."), 400
    return jsonify(check_message(text))


@app.get("/api/info")
def info():
    return jsonify(model_name=MODEL["model_name"], **MODEL["info"])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=False)
