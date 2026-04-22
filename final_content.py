from pymongo import MongoClient
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# =============================
# 공통 유틸
# =============================
def safe_cosine(a, b):
    a = np.asarray(a, dtype=float).reshape(1, -1)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    if not np.any(a) or not np.any(b):
        return 0.0
    v = float(cosine_similarity(a, b)[0, 0])
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return v

# =============================
# 데이터/모델 준비
# =============================
def prepare_song_vectors(mongo_uri="mongodb://localhost:27017/"):
    client = MongoClient(mongo_uri)
    db = client["music"]
    music_data = list(db["music"].find({}, {"_id": 0}))

    sentences = [song["lyrics"] for song in music_data if isinstance(song.get("lyrics"), list)]
    model = Word2Vec(sentences, vector_size=256, window=5, min_count=1, sg=1, epochs=10)

    all_genres = sorted(set(g for song in music_data for g in song.get("genre", [])))

    def get_lyrics_vector(lyrics):
        if not isinstance(lyrics, list) or len(lyrics) == 0:
            return np.zeros(model.vector_size, dtype=float)
        vectors = [model.wv[word] for word in lyrics if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size, dtype=float)

    def get_genre_vector(genre):
        return np.array([1.0 if g in (genre or []) else 0.0 for g in all_genres], dtype=float)

    song_vectors = {}
    for song in music_data:
        title = song.get("title")
        if not isinstance(title, str) or not title:
            continue
        lyrics_vec = get_lyrics_vector(song.get("lyrics", []))
        genre_vec = get_genre_vector(song.get("genre", []))
        song_vectors[title] = np.concatenate([lyrics_vec, genre_vec]).astype(float)

    return model, all_genres, song_vectors, music_data

# =============================
# 사용자 평균 벡터
# =============================
def compute_user_avg_vector(user_name, song_vectors, mongo_uri="mongodb://localhost:27017/"):
    client = MongoClient(mongo_uri)
    user_db = client["user"]

    if user_name not in user_db.list_collection_names():
        raise ValueError(f"사용자 {user_name}가 존재하지 않습니다.")

    collection = user_db[user_name]
    liked_titles = [doc.get("title") for doc in collection.find({}, {"_id": 0})]

    vectors = [song_vectors[t] for t in liked_titles if isinstance(t, str) and t in song_vectors]
    if not vectors:
        raise ValueError("유효한 곡 벡터가 없습니다. (좋아요 곡 벡터 없음)")

    avg_vector = np.mean(vectors, axis=0).astype(float)
    return avg_vector, liked_titles

# =============================
# 추천 (코사인 유사도 + 보너스 + per-artist 제한 + 연도 필터)
# =============================
def recommend_top_n_songs(
    user_vector,
    song_vectors,
    music_data,
    liked_titles,
    top_n=5,
    start_year=None,
    end_year=None,
    artist_bonus=0.5,
    genre_bonus=0.5,
    per_artist_cap=2,
    prefer_recent=True,
):
    def parse_year(year_raw):
        year = None
        if isinstance(year_raw, (int, float)):
            year = int(year_raw)
        elif isinstance(year_raw, str) and len(year_raw) >= 4 and year_raw[:4].isdigit():
            year = int(year_raw[:4])
        return year

    liked_titles_set = set(liked_titles)

    liked_artists = {
        s.get("artist")
        for s in music_data
        if s.get("title") in liked_titles_set and isinstance(s.get("artist"), str)
    }
    liked_genres = {
        g
        for s in music_data
        if s.get("title") in liked_titles_set
        for g in (s.get("genre") or [])
        if isinstance(g, str)
    }

    seen_titles = set()
    results = []

    for song in music_data:
        title = song.get("title")
        if not isinstance(title, str) or not title:
            continue
        if title in liked_titles_set or title in seen_titles:
            continue
        if title not in song_vectors:
            continue

        year = parse_year(song.get("release_year") or song.get("release_date") or "")
        if start_year is not None or end_year is not None:
            if year is None:
                continue
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue

        y = year if year is not None else -10**9

        song_vec = song_vectors[title]
        sim = safe_cosine(user_vector, song_vec)

        artist = song.get("artist", "")
        genre = song.get("genre", []) or []

        a_bonus = artist_bonus if (isinstance(artist, str) and artist in liked_artists) else 0.0
        g_bonus = genre_bonus if any((g in liked_genres) for g in genre if isinstance(g, str)) else 0.0

        final_score = sim + a_bonus + g_bonus

        results.append({
            "title": title,
            "artist": artist if isinstance(artist, str) else "unknown",
            "duration": song.get("duration", "--"),
            "youtube_url": song.get("youtube_url", ""),
            "final_score": float(final_score),
            "year": int(y),
        })
        seen_titles.add(title)

    results.sort(key=lambda x: x["final_score"], reverse=True)

    out = []
    artist_counts = {}
    for item in results:
        if len(out) >= top_n:
            break
        a = item.get("artist") or "unknown"
        if per_artist_cap and per_artist_cap > 0 and artist_counts.get(a, 0) >= per_artist_cap:
            continue
        out.append({
            "title": item["title"],
            "artist": item["artist"],
            "duration": item["duration"],
            "youtube_url": item["youtube_url"],
        })
        artist_counts[a] = artist_counts.get(a, 0) + 1

    return out
