from pymongo import MongoClient
import numpy as np
from gensim.models import Word2Vec
import re
from sklearn.metrics.pairwise import cosine_similarity

# ---------- 1) 텍스트 유틸 ----------
def make_string(text):
    return " ".join(text) if isinstance(text, list) else str(text)

def clean_lyrics(lyrics):
    lyrics = make_string(lyrics).lower()
    lyrics = re.sub(r"[^가-힣a-zA-Z\s]", "", lyrics)
    return lyrics

# ---------- 2) 키워드/불용어 ----------
stopwords = {
    "때","할때","가","은","는","이","의","에","를","을","도","과","와","한","또","좀","더","까지","만","으로","하고","에서",
    "듣기","좋은","노래","is","are","to","and","a","of","on","in","for","with","by","at","an","it","be","as","this","that"
}

def extract_keywords_simple(text):
    text = text.lower()
    text = re.sub(r"[^가-힣a-zA-Z\s]", "", text)
    words = text.split()
    return [w for w in words if w not in stopwords and len(w) >= 1]


# ---------- 3) 코사인 유틸 ----------
def cosine_sim(vec1, vec2):
    v1 = np.asarray(vec1, dtype=float).reshape(1, -1)
    v2 = np.asarray(vec2, dtype=float).reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0, 0])

# ---------- 4) 데이터 로딩 ----------
def music_data(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find({}, {"_id": 0}))
    print(f"불러온 곡 개수: {len(data)}")
    return data

# ---------- 5) Word2Vec 학습 ----------
def train_word2vec(music_data):
    lyrics_data = []
    for song in music_data:
        words = clean_lyrics(song.get("lyrics", "")).split()
        if words:
            lyrics_data.append(words)
    model = Word2Vec(lyrics_data, vector_size=300, window=10, min_count=1, workers=5, epochs=50)
    return model

# ---------- 6) 곡 벡터 구성 ----------
def build_song_vectors_from_model(model, music_data):
    song_vecs = {}
    for song in music_data:
        title = song.get("title")
        if not isinstance(title, str) or not title:
            continue
        words = clean_lyrics(song.get("lyrics", "")).split()
        vecs = [model.wv[w] for w in words if w in model.wv]
        song_vecs[title] = (np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size, dtype=float)).astype(float)
    return song_vecs

# ---------- 7) 사용자 평균 벡터 ----------
def get_user_avg_vector(user_name, song_vecs, uri="mongodb://localhost:27017/"):
    client = MongoClient(uri)
    user_db = client["user"]
    coll = user_db[user_name]
    liked_titles = [
        doc.get("title")
        for doc in coll.find({}, {"_id": 0})
        if isinstance(doc.get("title"), str)
    ]
    vecs = [song_vecs[t] for t in liked_titles if t in song_vecs]
    if not vecs:
        return None, liked_titles  # 좋아요는 있으나 벡터가 없을 때
    return np.mean(vecs, axis=0).astype(float), liked_titles

# ---------- 8) 유사도 계산(맥락/단어레벨) ----------
def average_vector_similarity(query_vecs, lyrics_vecs):  # 키워드와 가사의 맥락 유사도
    if not query_vecs or not lyrics_vecs:
        return 0.0
    q_mean = np.mean(query_vecs, axis=0)
    l_mean = np.mean(lyrics_vecs, axis=0)
    return cosine_sim(q_mean, l_mean)

def word_level_similarity(query_keywords, lyrics, model, top_k=3, max_lyrics_words=400):  # 세부 유사도
    lyrics_words = clean_lyrics(lyrics).split()[:max_lyrics_words]
    lyrics_vecs = [model.wv[w] for w in lyrics_words if w in model.wv]
    query_vecs  = [model.wv[w] for w in query_keywords if w in model.wv]
    if not lyrics_vecs or not query_vecs:
        return 0.0

    sims = []
    for qv in query_vecs:
        scores = [cosine_sim(qv, lv) for lv in lyrics_vecs]
        scores.sort(reverse=True)
        sims.append(np.mean(scores[:top_k]))
    word_topk = np.mean(sims)
    avg_sim   = average_vector_similarity(query_vecs, lyrics_vecs)
    return 0.6 * word_topk + 0.4 * avg_sim

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).casefold()

def recommend_by_query_words_personalized(
    user_name,
    query_keywords,
    songs,
    model,
    song_vecs,
    top_n=5,
    w_query=0.8,
    w_user=0.2,
    per_kw_boost=0.02,
    max_boost=None,
    start_year=None,
    end_year=None,
    prefer_recent=True
):
    # 1) 사용자 평균 벡터
    try:
        user_vec, liked_titles = get_user_avg_vector(user_name, song_vecs)
    except Exception:
        user_vec, liked_titles = (None, [])
    liked_set = {_norm(t) for t in liked_titles}

    # 2) 키워드: list 유지(전부 반영)
    kws_list = [ _norm(k) for k in query_keywords if isinstance(k, str) and k.strip() ]
    kws_list = [k for k in kws_list if k]

    # 3) 연도 파서
    def parse_year(year_raw):
        year = None
        if isinstance(year_raw, (int, float)):
            year = int(year_raw)
        elif isinstance(year_raw, str) and len(year_raw) >= 4 and year_raw[:4].isdigit():
            year = int(year_raw[:4])
        return year

    results = []
    seen = set()

    for song in songs:
        title = song.get("title", "")
        if not isinstance(title, str) or not title:
            continue
        if _norm(title) in liked_set:
            continue

        artist = song.get("artist", "알 수 없음")
        dup_key = (_norm(title), _norm(artist))
        if dup_key in seen:
            continue
        seen.add(dup_key)

        # 4) 연도 필터(점수에는 영향 X)
        year = parse_year(song.get("release_year") or song.get("release_date") or "")
        if start_year is not None or end_year is not None:
            if year is None:
                continue
            if start_year is not None and year < int(start_year):
                continue
            if end_year is not None and year > int(end_year):
                continue
        y_for_sort = year if year is not None else 0  # 정렬용 보조키

        # 5) 테마 점수(키워드 vs 가사)
        theme_score = float(
            word_level_similarity(kws_list, song.get("lyrics", ""), model) or 0.0
        )

        # 6) 개인화 점수(사용자 평균 ↔ 곡 벡터)
        user_score = 0.0
        if user_vec is not None and title in song_vecs:
            sim = cosine_sim(user_vec, song_vecs[title])
            if not np.isfinite(sim):
                sim = 0.0
            user_score = (sim + 1.0) / 2.0  # [-1,1] → [0,1]

        title_text  = _norm(song.get("title", ""))
        genre_raw   = song.get("genre", [])
        genre_text  = _norm(" ".join(genre_raw) if isinstance(genre_raw, list) else str(genre_raw))

        boost = 0.0
        for kw in kws_list:
            if (kw in title_text) or (kw in genre_text):
                boost += per_kw_boost

        final_score = w_query * theme_score + w_user * user_score + boost

        results.append((
            float(final_score),
            int(y_for_sort),
            {
                "title": title,
                "artist": artist,
                "duration": song.get("duration", "--"),
                "youtube_url": song.get("youtube_url", "")
            }
        ))

    # 9) 정렬: prefer_recent=True면 (점수, 연도) 동시 내림차순
    if prefer_recent:
        results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    else:
        results.sort(key=lambda x: x[0], reverse=True)

    return [item for _, _, item in results[:max(0, int(top_n))]]