import os
import joblib
import numpy as np
import pandas as pd
import requests

# ─── TMDB API Config ─────────────────────────────────────────────────────────
TMDB_API_KEY = "c80580c88e36c920b2d907c961767396"
TMDB_BASE    = "https://api.themoviedb.org/3"

# Load all the required components
OUTPUT_DIR = "cinepredict_outputs"

print("Loading models and encoders...")
xgb_verdict = joblib.load(os.path.join(OUTPUT_DIR, "xgb_verdict_classifier.pkl"))
xgb_revenue = joblib.load(os.path.join(OUTPUT_DIR, "xgb_revenue_regressor.pkl"))
xgb_rating  = joblib.load(os.path.join(OUTPUT_DIR, "xgb_rating_regressor.pkl"))
le_genre    = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder_genre.pkl"))
le_verdict  = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder_verdict.pkl"))

ALL_RAW_FEATURES = [
    "budget", "budget_log", "release_year", "release_month", "num_genres",
    "director_avg_past_rating", "lead_actor_avg_rating", "sidecast_avg_rating", "genre_trend_score",
    "is_action", "is_comedy", "is_drama", "is_horror", "is_sciencefiction", "is_animation", "is_romance", "is_thriller",
    "primary_genre"
]

# ─── TMDB Helper Functions ────────────────────────────────────────────────────

def search_person(name: str) -> dict | None:
    """Search TMDB for a person by name, return the best match result dict or None."""
    url = f"{TMDB_BASE}/search/person"
    resp = requests.get(url, params={"api_key": TMDB_API_KEY, "query": name}, timeout=10)
    if resp.status_code != 200:
        return None
    results = resp.json().get("results", [])
    if not results:
        return None
    # Return the most popular result
    return sorted(results, key=lambda x: x.get("popularity", 0), reverse=True)[0]


def get_avg_rating_for_person(person_id: int, department: str, before_year: int) -> float | None:
    """
    Fetch all movies for a person and compute the average TMDB vote_average
    for movies released strictly before `before_year`.
    department: 'Directing' for directors, 'Acting' for actors.
    """
    url = f"{TMDB_BASE}/person/{person_id}/movie_credits"
    resp = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=10)
    if resp.status_code != 200:
        return None

    data = resp.json()
    # crew = directing credits, cast = acting credits
    if department == "Directing":
        credits = [c for c in data.get("crew", []) if c.get("job") == "Director"]
    else:
        credits = data.get("cast", [])

    ratings = []
    for movie in credits:
        release = movie.get("release_date", "")
        vote    = movie.get("vote_average", 0)
        count   = movie.get("vote_count", 0)
        if not release or vote == 0 or count < 50:
            continue
        try:
            year = int(release[:4])
        except ValueError:
            continue
        if year < before_year:
            ratings.append(vote)

    if not ratings:
        return None
    return round(sum(ratings) / len(ratings), 2)


def fetch_person_rating(prompt_label: str, department: str, before_year: int, default: float = 6.0) -> tuple[float, str]:
    """
    Ask the user for a name, look them up on TMDB, and return their avg past rating.
    Returns (rating, name_used).
    """
    name = input(f"{prompt_label} name: ").strip()
    if not name:
        print(f"  → No name entered. Using default rating of {default}.")
        return default, "Unknown"

    print(f"  → Searching TMDB for '{name}'...")
    person = search_person(name)
    if not person:
        print(f"  → Could not find '{name}' on TMDB. Using default rating of {default}.")
        return default, name

    pid   = person["id"]
    found = person["name"]
    print(f"  → Found: {found} (popularity: {person.get('popularity', 0):.1f})")

    avg = get_avg_rating_for_person(pid, department, before_year)
    if avg is None:
        print(f"  → Not enough past film data before {before_year}. Using default rating of {default}.")
        return default, found

    # TMDB ratings are out of 10, same scale as IMDb — no conversion needed
    print(f"  → Average past rating (pre-{before_year}): {avg} / 10")
    return avg, found


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_movie(movie_data):
    """Takes a dictionary of movie metadata, processes it, and returns predictions."""
    df = pd.DataFrame([movie_data], columns=ALL_RAW_FEATURES)

    try:
        df["primary_genre"] = le_genre.transform(df["primary_genre"])
    except ValueError:
        print(f"Warning: Genre '{movie_data['primary_genre']}' unseen in training. Defaulting to first known.")
        df["primary_genre"] = 0

    # 1. Verdict
    verdict_pred_encoded = xgb_verdict.predict(df)[0]
    verdict_pred         = le_verdict.inverse_transform([verdict_pred_encoded])[0]
    verdict_probs        = xgb_verdict.predict_proba(df)[0]
    confidence_score     = verdict_probs[verdict_pred_encoded] * 100

    # 2. Revenue
    revenue_log_pred = xgb_revenue.predict(df)[0]
    revenue_pred     = np.expm1(revenue_log_pred)

    # 3. Rating
    rating_pred = xgb_rating.predict(df)[0]

    return {
        "Verdict":               f"{verdict_pred} (Confidence: {confidence_score:.1f}%)",
        "Box Office Revenue (USD)": f"${revenue_pred:,.2f}",
        "IMDb Rating":           f"{rating_pred:.1f} / 10"
    }


# ─── User Input ───────────────────────────────────────────────────────────────

def get_user_input():
    print("\n--- Enter Movie Details ---")
    try:
        budget        = float(input("Budget (in USD, e.g., 150000000): ").replace(",", ""))
        budget_log    = np.log1p(budget)
        release_year  = int(input("Release Year (e.g., 2024): "))
        release_month = int(input("Release Month (1-12): "))
        num_genres    = int(input("Number of Genres: "))

        # ── Director (TMDB lookup) ────────────────────────────────────────────
        print()
        director_avg_past_rating, _ = fetch_person_rating(
            "Director", "Directing", release_year
        )

        # ── Lead Actor (TMDB lookup) ──────────────────────────────────────────
        print()
        lead_actor_avg_rating, _ = fetch_person_rating(
            "Lead Actor", "Acting", release_year
        )

        # ── Supporting Cast (TMDB lookup, average of up to 2 names) ──────────
        print()
        print("Supporting cast (up to 2 names for sidecast rating):")
        sidecast_ratings = []
        for i in range(1, 3):
            rating, _ = fetch_person_rating(
                f"  Supporting Actor {i} (leave blank to skip)", "Acting", release_year
            )
            sidecast_ratings.append(rating)
        sidecast_avg_rating = round(sum(sidecast_ratings) / len(sidecast_ratings), 2)
        print(f"  → Sidecast average: {sidecast_avg_rating} / 10")

        # ── Genre Trend ───────────────────────────────────────────────────────
        print("\nHow popular is this genre right now? (Select 1-3)")
        print(" 1. Cold/Declining (Audiences are tired of it)")
        print(" 2. Neutral/Average (Standard performance)")
        print(" 3. Hot/Booming (Extremely popular right now)")
        trend_choice = input("Enter 1, 2, or 3 [Default 2]: ").strip()

        if trend_choice == "1":
            genre_trend_score = -50000000.0
        elif trend_choice == "3":
            genre_trend_score = 100000000.0
        else:
            genre_trend_score = 0.0

        # ── Genre Flags ───────────────────────────────────────────────────────
        print("\nSpecify Genres (1 for Yes, 0 for No):")
        is_action       = int(input("Is Action? "))
        is_comedy       = int(input("Is Comedy? "))
        is_drama        = int(input("Is Drama? "))
        is_horror       = int(input("Is Horror? "))
        is_sciencefiction = int(input("Is Science Fiction? "))
        is_animation    = int(input("Is Animation? "))
        is_romance      = int(input("Is Romance? "))
        is_thriller     = int(input("Is Thriller? "))

        primary_genre = input("\nPrimary Genre (e.g., Action, Comedy): ")

        return {
            "budget":                     budget,
            "budget_log":                 budget_log,
            "release_year":               release_year,
            "release_month":              release_month,
            "num_genres":                 num_genres,
            "director_avg_past_rating":   director_avg_past_rating,
            "lead_actor_avg_rating":      lead_actor_avg_rating,
            "sidecast_avg_rating":        sidecast_avg_rating,
            "genre_trend_score":          genre_trend_score,
            "is_action":                  is_action,
            "is_comedy":                  is_comedy,
            "is_drama":                   is_drama,
            "is_horror":                  is_horror,
            "is_sciencefiction":          is_sciencefiction,
            "is_animation":               is_animation,
            "is_romance":                 is_romance,
            "is_thriller":                is_thriller,
            "primary_genre":              primary_genre
        }

    except ValueError:
        print("Invalid input. Please enter appropriate numeric values where required.")
        return None


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    movie = get_user_input()

    if not movie:
        print("Prediction aborted due to invalid input.")
        exit(1)

    print("\nPredicting for Movie with following specs:")
    for key, value in movie.items():
        print(f"  {key}: {value}")

    print("\n--- Predictions ---")
    predictions = predict_movie(movie)
    for key, val in predictions.items():
        print(f"{key}: {val}")
