import os
import sys
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ─── TMDB Config ──────────────────────────────────────────────────────────────
TMDB_API_KEY = "c80580c88e36c920b2d907c961767396"
TMDB_BASE    = "https://api.themoviedb.org/3"

# ─── Load Models ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "cinepredict_outputs"
xgb_verdict = joblib.load(os.path.join(OUTPUT_DIR, "xgb_verdict_classifier.pkl"))
xgb_revenue = joblib.load(os.path.join(OUTPUT_DIR, "xgb_revenue_regressor.pkl"))
xgb_rating  = joblib.load(os.path.join(OUTPUT_DIR, "xgb_rating_regressor.pkl"))
le_genre    = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder_genre.pkl"))
le_verdict  = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder_verdict.pkl"))

ALL_RAW_FEATURES = [
    "budget", "budget_log", "release_year", "release_month", "num_genres",
    "director_avg_past_rating", "lead_actor_avg_rating", "sidecast_avg_rating", "genre_trend_score",
    "is_action", "is_comedy", "is_drama", "is_horror", "is_sciencefiction",
    "is_animation", "is_romance", "is_thriller", "primary_genre"
]

# ─── TMDB Helpers ─────────────────────────────────────────────────────────────
def search_person(name):
    try:
        resp = requests.get(f"{TMDB_BASE}/search/person",
                            params={"api_key": TMDB_API_KEY, "query": name}, timeout=8)
        results = resp.json().get("results", [])
        if not results:
            return None
        return sorted(results, key=lambda x: x.get("popularity", 0), reverse=True)[0]
    except Exception:
        return None

def get_avg_rating(person_id, department, before_year):
    try:
        resp = requests.get(f"{TMDB_BASE}/person/{person_id}/movie_credits",
                            params={"api_key": TMDB_API_KEY}, timeout=8)
        data = resp.json()
        credits = ([c for c in data.get("crew", []) if c.get("job") == "Director"]
                   if department == "Directing" else data.get("cast", []))
        ratings = []
        for m in credits:
            rel = m.get("release_date", "")
            vote = m.get("vote_average", 0)
            count = m.get("vote_count", 0)
            if not rel or vote == 0 or count < 50:
                continue
            try:
                if int(rel[:4]) < before_year:
                    ratings.append(vote)
            except ValueError:
                continue
        return round(sum(ratings) / len(ratings), 2) if ratings else None
    except Exception:
        return None

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/lookup-person", methods=["POST"])
def lookup_person():
    """Given a name + department + year, return their avg past rating from TMDB."""
    data       = request.json
    name       = data.get("name", "").strip()
    department = data.get("department", "Acting")
    before_year = int(data.get("before_year", 2024))

    if not name:
        return jsonify({"error": "No name provided"}), 400

    person = search_person(name)
    if not person:
        return jsonify({"error": f"Could not find '{name}' on TMDB", "rating": 6.0, "found_name": name})

    pid        = person["id"]
    found_name = person["name"]
    profile    = person.get("profile_path")
    popularity = person.get("popularity", 0)

    avg = get_avg_rating(pid, department, before_year)
    rating = avg if avg is not None else 6.0

    return jsonify({
        "found_name":  found_name,
        "rating":      rating,
        "popularity":  round(popularity, 1),
        "profile_path": f"https://image.tmdb.org/t/p/w185{profile}" if profile else None,
        "data_found":  avg is not None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.json
    try:
        genre_trend_map = {1: -50000000.0, 2: 0.0, 3: 100000000.0}
        genre_trend_score = genre_trend_map.get(int(body.get("genre_trend", 2)), 0.0)

        budget = float(body["budget"])
        movie_data = {
            "budget":                   budget,
            "budget_log":               np.log1p(budget),
            "release_year":             int(body["release_year"]),
            "release_month":            int(body["release_month"]),
            "num_genres":               int(body["num_genres"]),
            "director_avg_past_rating": float(body["director_rating"]),
            "lead_actor_avg_rating":    float(body["lead_actor_rating"]),
            "sidecast_avg_rating":      float(body["sidecast_rating"]),
            "genre_trend_score":        genre_trend_score,
            "is_action":                int(body.get("is_action", 0)),
            "is_comedy":                int(body.get("is_comedy", 0)),
            "is_drama":                 int(body.get("is_drama", 0)),
            "is_horror":                int(body.get("is_horror", 0)),
            "is_sciencefiction":        int(body.get("is_sciencefiction", 0)),
            "is_animation":             int(body.get("is_animation", 0)),
            "is_romance":               int(body.get("is_romance", 0)),
            "is_thriller":              int(body.get("is_thriller", 0)),
            "primary_genre":            body.get("primary_genre", "Action"),
        }

        df = pd.DataFrame([movie_data], columns=ALL_RAW_FEATURES)
        try:
            df["primary_genre"] = le_genre.transform(df["primary_genre"])
        except ValueError:
            df["primary_genre"] = 0

        verdict_enc    = xgb_verdict.predict(df)[0]
        verdict        = le_verdict.inverse_transform([verdict_enc])[0]
        probs          = xgb_verdict.predict_proba(df)[0]
        confidence     = float(probs[verdict_enc] * 100)
        all_probs      = {le_verdict.inverse_transform([i])[0]: float(p * 100) for i, p in enumerate(probs)}

        revenue        = float(np.expm1(xgb_revenue.predict(df)[0]))
        imdb_rating    = float(xgb_rating.predict(df)[0])

        return jsonify({
            "verdict":    verdict,
            "confidence": round(confidence, 1),
            "all_probs":  all_probs,
            "revenue":    round(revenue, 2),
            "imdb_rating": round(imdb_rating, 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
