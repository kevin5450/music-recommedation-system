from content_filtering import prepare_song_vectors, compute_user_avg_vector
from collaborative_filtering import load_user_likes
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# =============================
# 공통 유틸
# =============================
def safe_cosine(a, b):
    a = np.asarray(a, dtype=float).reshape(1, -1)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    if not np.any(a) or not np.any(b):
        return 0.0
    v = float(cosine_similarity(a, b)[0, 0])
    return 0.0 if (not np.isfinite(v)) else v

# =============================
# 하이브리드 추천 (콘텐츠+협업, 연도 필터)
# =============================
def recommend_hybrid_weighted(
    user_name,
    mongo_uri="mongodb://localhost:27017/",
    content_weight=0.6,
    collab_weight=0.4,
    start_year=None,
    end_year=None,
):
    _, _, song_vectors, music_data = prepare_song_vectors(mongo_uri)
    user_vector, liked_titles = compute_user_avg_vector(user_name, song_vectors, mongo_uri)

    def apply_year_filter(music_data, start_year, end_year):
        if start_year is None and end_year is None:
            return {s.get("title"): s for s in music_data if isinstance(s.get("title"), str)}
        filtered = {}
        for s in music_data:
            title = s.get("title")
            if not isinstance(title, str):
                continue
            yr_raw = s.get("release_year") or s.get("release_date") or ""
            year = None
            if isinstance(yr_raw, (int, float)):
                year = int(yr_raw)
            elif isinstance(yr_raw, str) and len(yr_raw) >= 4 and yr_raw[:4].isdigit():
                year = int(yr_raw[:4])
            else:
                year = None
            if year is None:
                continue
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue
            filtered[title] = s
        return filtered

    music_dict = apply_year_filter(music_data, start_year, end_year)

    content_scores = {}
    uv = user_vector.reshape(1, -1)
    for title, vec in song_vectors.items():
        if title in liked_titles or title not in music_dict:
            continue
        content_scores[title] = safe_cosine(uv, vec)

    likes_res = load_user_likes(mongo_uri)
    if isinstance(likes_res, tuple) and len(likes_res) == 2:
        user_ids, user_likes = likes_res
    else:
        user_likes = likes_res
        user_ids = list(user_likes.keys())

    if user_name not in user_ids:
        raise ValueError(f"사용자 {user_name}가 존재하지 않습니다.")

    filtered_titles = list(music_dict.keys())
    user_index = {uid: idx for idx, uid in enumerate(user_ids)}
    title_index = {title: idx for idx, title in enumerate(filtered_titles)}

    if len(user_ids) == 0 or len(filtered_titles) == 0:
        collab_scores = {}
    else:
        matrix = np.zeros((len(user_ids), len(filtered_titles)), dtype=float)
        for uid, likes in user_likes.items():
            ui = user_index[uid]
            for t in likes:
                j = title_index.get(t)
                if j is not None:
                    matrix[ui, j] = 1.0

        target_idx = user_index[user_name]
        target_vec = matrix[target_idx].reshape(1, -1)
        if np.any(target_vec):
            user_sims = cosine_similarity(target_vec, matrix)[0]
        else:
            user_sims = np.zeros(matrix.shape[0], dtype=float)
        if 0 <= target_idx < user_sims.shape[0]:
            user_sims[target_idx] = 0.0
        user_sims = np.clip(user_sims, 0.0, None)

        weighted = matrix * user_sims.reshape(-1, 1)
        item_scores = np.sum(weighted, axis=0) if weighted.size else np.zeros(len(filtered_titles), dtype=float)

        collab_scores = {}
        for j, title in enumerate(filtered_titles):
            if title in liked_titles:
                continue
            val = float(item_scores[j]) if j < len(item_scores) else 0.0
            collab_scores[title] = 0.0 if (not np.isfinite(val)) else val

    final_titles = set(content_scores.keys()) | set(collab_scores.keys())
    results = []
    for title in final_titles:
        song = music_dict.get(title)
        if not song:
            continue
        content_score = content_scores.get(title, 0.0)
        collab_score = collab_scores.get(title, 0.0)
        final_score = content_weight * content_score + collab_weight * collab_score
        if not np.isfinite(final_score):
            final_score = 0.0
        results.append({
            "title": title,
            "artist": song.get("artist", "unknown"),
            "duration": song.get("duration", "--"),
            "youtube_url": song.get("youtube_url", ""),
            "final_score": float(final_score),
        })

    if not results:
        return []
    results = [r for r in results if "final_score" in r and np.isfinite(r["final_score"])]
    results.sort(key=lambda x: x.get("final_score", -1e9), reverse=True)
    return results[:5]
