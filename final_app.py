from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient, ASCENDING
from threading import Lock
import traceback
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import time


import os, csv, datetime

# ===== 우리 모듈 =====
from final_content import (
    prepare_song_vectors,      
    compute_user_avg_vector,
    recommend_top_n_songs
)
from final_collaborative import (
    load_user_likes,
    recommend_user_cf_overlap_proportional as recommend_user_cf_overlap
)
from final_theme_based_filtering import (
    recommend_by_query_words_personalized,
    extract_keywords_simple,
)

# =============================
# 공통 상수/유틸
# =============================
MONGO_URI = "mongodb://localhost:27017/"

def safe_cosine(a, b):
    a = np.asarray(a, dtype=float).reshape(1, -1)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    if not np.any(a) or not np.any(b):
        return 0.0
    v = float(cosine_similarity(a, b)[0, 0])
    return 0.0 if not np.isfinite(v) else v

def extract_year_from_date(date_str):
    if not date_str:
        return None
    try:
        y = int(date_str)
        if 1900 <= y <= 2030:
            return y
    except (ValueError, TypeError):
        pass
    if isinstance(date_str, str):
        m = re.match(r'^(\d{4})[.\-]', date_str.strip())
        if m:
            try:
                y = int(m.group(1))
                if 1900 <= y <= 2030:
                    return y
            except ValueError:
                pass
    return None

def _parse_year_ranges(args):
    starts = args.getlist("start_year")
    ends   = args.getlist("end_year")
    ranges = []
    n = min(len(starts), len(ends))
    for i in range(n):
        try:
            a = int(starts[i]); b = int(ends[i])
        except (ValueError, TypeError):
            continue
        if a > b:
            a, b = b, a
        ranges.append((a, b))
    return ranges

def _year_in_ranges(year, ranges):
    if not ranges:
        return True
    if year is None:
        return False
    for a, b in ranges:
        if a <= year <= b:
            return True
    return False

def _filter_music_by_ranges(music_data, ranges):
    if not ranges:
        return music_data
    out = []
    for s in music_data:
        y = extract_year_from_date(s.get("release_year") or s.get("release_date"))
        if _year_in_ranges(y, ranges):
            out.append(s)
    return out

# =============================
# 앱/캐시
# =============================
app = Flask(__name__)
CORS(app)
#app.config['JSON_AS_ASCII'] = False
MUSIC_DATA = None
SONG_VECTORS = None
USER_IDS = None
USER_LIKES = None
THEME_MODEL = None  # <- prepare_song_vectors()가 만든 Word2Vec 모델 재사용

TITLE_INDEX = None
ALL_TITLES = None
USER_INDEX = None
USER_ITEM_MATRIX = None

CACHE_LOCK = Lock()

# [ADD] CSV 경로/헤더
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "precision_log.csv")
CSV_FIELDS = [
    "timestamp","user","content","collabo","hybrid","theme",
    "score_content","score_collabo","score_hybrid","score_theme", 
    "items_content","genres_content",
    "items_collabo","genres_collabo",
    "items_hybrid","genres_hybrid",
    "items_theme","genres_theme","theme_keywords"
]

def _ensure_csv_header():
    need_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    if need_header:
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

def _norm_str(x):
    if x is None: return ""
    return str(x)

def _titles_and_genres(items):
    """items: [{'title','artist','genre',...}, ...]"""
    titles, genres = [], []
    for it in items or []:
        t = _norm_str(it.get("title","")).strip()
        if t: titles.append(t)
        g = it.get("genre", "")
        if isinstance(g, list):
            for v in g:
                vs = _norm_str(v).strip()
                if vs: genres.append(vs)
        else:
            gstr = _norm_str(g).strip()
            if gstr:
                parts = [p.strip() for p in gstr.split(",")]
                genres.extend([p for p in parts if p])
    genres = sorted(set(genres))
    return titles, genres

def _precision(items, total):
    try:
        total = int(total) if total is not None else 0
    except:
        total = 0
    sel = len(items or [])
    return (sel / total) if total > 0 else 0.0

def _enrich_items_with_db(items):
    """클라이언트에서 genre가 비어왔을 때 music DB로 보강"""
    if not items: return []
    client = MongoClient(MONGO_URI)
    col_music = client["music"]["music"]
    out = []
    for it in items:
        merged = dict(it)
        title = _norm_str(it.get("title")).strip()
        if title:
            doc = col_music.find_one({"title": title}, {"_id":0, "genre":1, "duration":1, "youtube_url":1, "artist":1})
            if doc:
                for k in ["genre","duration","youtube_url","artist"]:
                    if not merged.get(k) and doc.get(k) is not None:
                        merged[k] = doc.get(k)
        out.append(merged)
    return out


# =============================
# 초기화
# =============================
def init_all():
    global MUSIC_DATA, SONG_VECTORS, USER_IDS, USER_LIKES
    global THEME_MODEL
    global TITLE_INDEX, ALL_TITLES, USER_INDEX, USER_ITEM_MATRIX

    with CACHE_LOCK:
        try:
            # final_content.prepare_song_vectors는 (model, all_genres, song_vectors, music_data) 반환
            _model_tmp, _all_genres, SONG_VECTORS, MUSIC_DATA = prepare_song_vectors(MONGO_URI)
            THEME_MODEL = _model_tmp  # <- 테마용 모델로 재사용
            print(f"[초기화] 음악 데이터 로드 완료: {len(MUSIC_DATA)}곡")
            print(f"[초기화] 곡 벡터 준비 완료: {len(SONG_VECTORS)}개")
            print(f"[초기화] 테마 모델 준비 완료: dim={THEME_MODEL.vector_size}")

            # 협업 필터링용 사용자-아이템 준비
            likes_res = load_user_likes(MONGO_URI)
            if isinstance(likes_res, tuple) and len(likes_res) == 2:
                USER_IDS, USER_LIKES = likes_res
            elif isinstance(likes_res, dict):
                USER_LIKES = likes_res
                USER_IDS = list(likes_res.keys())
            else:
                USER_IDS, USER_LIKES = [], {}
            print(f"[초기화] 사용자 데이터 로드 완료: {len(USER_IDS)}명")

            ALL_TITLES = sorted({s.get("title") for s in MUSIC_DATA if isinstance(s.get("title"), str)})
            TITLE_INDEX = {t: i for i, t in enumerate(ALL_TITLES)}
            USER_INDEX = {u: i for i, u in enumerate(USER_IDS)}

            n_u, n_i = len(USER_IDS), len(ALL_TITLES)
            USER_ITEM_MATRIX = np.zeros((n_u, n_i), dtype=float)
            for uid, likes in USER_LIKES.items():
                ui = USER_INDEX.get(uid)
                if ui is None:
                    continue
                for t in likes:
                    tj = TITLE_INDEX.get(t)
                    if tj is not None:
                        USER_ITEM_MATRIX[ui, tj] = 1.0

            print(f"[초기화] 완료 - 음악 {len(MUSIC_DATA)}곡, 사용자 {len(USER_IDS)}명, "
                  f"벡터 {len(SONG_VECTORS)}개, 행렬 {USER_ITEM_MATRIX.shape}")

        except Exception as e:
            print(f"[ERROR] 초기화 실패: {e}")
            traceback.print_exc()

# =============================
# 1) 콘텐츠 기반 추천
# =============================
@app.get("/recommend/content")
def recommend_content():
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "사용자 이름이 없습니다."}), 400
    try:
        ranges = _parse_year_ranges(request.args)
        filtered = _filter_music_by_ranges(MUSIC_DATA, ranges)

        user_vector, liked_titles = compute_user_avg_vector(user, SONG_VECTORS, MONGO_URI)
        recs = recommend_top_n_songs(user_vector, SONG_VECTORS, filtered, liked_titles)
        return jsonify({"user": user, "ranges": ranges, "recommendations": recs})
    except Exception as e:
        print("[ERROR][content]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =============================
# 2) 협업 필터링 추천
# =============================
@app.get("/recommend/collab")
def recommend_collab():
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "사용자 이름이 없습니다."}), 400
    try:
        if not isinstance(USER_LIKES, dict) or user not in USER_LIKES:
            return jsonify({"error": "사용자 데이터가 없습니다."}), 400

        ranges = _parse_year_ranges(request.args)
        filtered = _filter_music_by_ranges(MUSIC_DATA, ranges)
        allowed = {s.get("title") for s in filtered if isinstance(s.get("title"), str)}

        result = recommend_user_cf_overlap(user, USER_LIKES, top_n=5, music_data=filtered)
        result = [r for r in result if (isinstance(r, dict) and r.get("title") in allowed)][:5]
        return jsonify({"user": user, "ranges": ranges, "recommendations": result})
    except Exception as e:
        print("[ERROR][collab]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =============================
# 3) 테마 기반 추천
# =============================
@app.get("/recommend/theme")
def recommend_theme():
    t0 = time.perf_counter()
    log_ok = False
    try:
        user  = (request.args.get("user")  or "").strip()
        query = (request.args.get("query") or "").strip()
        if not user:
            resp = jsonify({"error": "user 파라미터가 필요합니다."}), 400
            return resp

        ranges = _parse_year_ranges(request.args)
        filtered = _filter_music_by_ranges(MUSIC_DATA, ranges) if ranges else MUSIC_DATA

        if THEME_MODEL is None:
            user_vector, liked_titles = compute_user_avg_vector(user, SONG_VECTORS, MONGO_URI)
            recs = recommend_top_n_songs(
                user_vector=user_vector,
                song_vectors=SONG_VECTORS,
                music_data=filtered,
                liked_titles=liked_titles,
                top_n=5
            )
            payload = {
                "user": user, "query": query, "keywords": [], "ranges": ranges,
                "mode": "fallback_content", "recommendations": recs
            }
        else:
            keywords = extract_keywords_simple(query) if query else []
            recs = recommend_by_query_words_personalized(
                user_name=user,
                query_keywords=keywords,
                songs=filtered,
                model=THEME_MODEL,
                song_vecs=SONG_VECTORS,
                top_n=5,
                w_query=0.8, w_user=0.2, per_kw_boost=0.10, max_boost=0.30,
                start_year=None, end_year=None, prefer_recent=True
            )

            out = []
            for i, item in enumerate(recs, start=1):
                out.append({
                    "rank": i,
                    "title": item.get("title"),
                    "artist": item.get("artist", "unknown"),
                    "duration": item.get("duration", "--"),
                    "youtube_url": item.get("youtube_url", ""),
                    "genre": item.get("genre", [])
                })

            payload = {"user": user, "query": query, "keywords": keywords, "ranges": ranges, "recommendations": out}

        log_ok = True
        return jsonify(payload)

    finally:
        elapsed = time.perf_counter() - t0
        app.logger.info("[DEBUG] /recommend/theme 실행 시간: %.4f 초 (logged=%s)", elapsed, log_ok)

# =============================
# 4) 하이브리드 추천
# =============================
@app.get("/recommend/hybrid")
def hybrid_recommendation():
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "사용자 이름이 없습니다."}), 400
    try:
        ranges = _parse_year_ranges(request.args)
        filtered = _filter_music_by_ranges(MUSIC_DATA, ranges)
        allowed_titles = {s.get("title") for s in filtered if isinstance(s.get("title"), str)}

        user_vector, liked_titles = compute_user_avg_vector(user, SONG_VECTORS, MONGO_URI)
        content_scores = {}
        for title, svec in SONG_VECTORS.items():
            if title in liked_titles or title not in allowed_titles:
                continue
            try:
                content_scores[title] = safe_cosine(user_vector, svec)
            except Exception:
                content_scores[title] = 0.0

        if user not in USER_INDEX:
            return jsonify({"error": "사용자 데이터가 없습니다."}), 400

        collab_scores = {}
        if USER_ITEM_MATRIX.size != 0 and len(USER_IDS) > 1:
            tgt_idx = USER_INDEX[user]
            norms = np.linalg.norm(USER_ITEM_MATRIX, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            U_norm = USER_ITEM_MATRIX / norms
            sims = (U_norm[tgt_idx : tgt_idx + 1, :] @ U_norm.T).ravel()
            if sims.size > tgt_idx:
                sims[tgt_idx] = 0.0
            sims = np.clip(sims, 0.0, None)
            item_scores = sims.reshape(1, -1) @ USER_ITEM_MATRIX
            item_scores = item_scores.ravel()

            title_to_idx = {t: j for j, t in enumerate(ALL_TITLES)}
            for title in allowed_titles:
                j = title_to_idx.get(title)
                if j is None or title in liked_titles:
                    continue
                val = float(item_scores[j])
                collab_scores[title] = val if np.isfinite(val) else 0.0

        results = []
        final_titles = set(content_scores.keys()) | set(collab_scores.keys())
        for title in final_titles:
            song = next((s for s in filtered if s.get("title") == title), None)
            if not song:
                continue
            fs = 0.6 * content_scores.get(title, 0.0) + 0.4 * collab_scores.get(title, 0.0)
            if not np.isfinite(fs):
                fs = 0.0
            results.append({
                "title": title,
                "artist": song.get("artist", "unknown"),
                "duration": song.get("duration", "--"),
                "youtube_url": song.get("youtube_url", ""),
                "genre": song.get("genre", []),
                "final_score": float(fs),
                "content_cos": float(content_scores.get(title, 0.0)),
                "collab_cos": float(collab_scores.get(title, 0.0)),
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return jsonify({"user": user, "ranges": ranges, "recommendations": results[:5]})
    except Exception as e:
        print("[ERROR][hybrid]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =============================
# 피드백(개별 - 기존)
# =============================
@app.post("/feedback/like")
def feedback_like():
    data = request.get_json(silent=True) or {}
    user = (data.get("user") or "").strip()
    items = data.get("items") or []
    if not user:
        return jsonify({"error": "user가 비어 있습니다."}), 400
    try:
        client = MongoClient(MONGO_URI)
        db_user = client["user"]
        coll_likes = db_user[user]
        coll_likes.create_index([("title", ASCENDING), ("artist", ASCENDING)], unique=True)

        inserted, updated = 0, 0
        for it in items:
            title = (it.get("title") or "").strip()
            artist = (it.get("artist") or "").strip()
            if not title or not artist:
                continue
            res = coll_likes.update_one(
                {"title": title, "artist": artist},
                {"$set": {"title": title, "artist": artist, "liked": True, "rating": 3}},
                upsert=True
            )
            if res.upserted_id:
                inserted += 1
            else:
                updated += 1
        return jsonify({"message": f"likes 저장 완료 (추가 {inserted}, 갱신 {updated})"})
    except Exception as e:
        print("[ERROR][feedback/like]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.post("/feedback/overall")
def feedback_overall():
    data = request.get_json(silent=True) or {}
    user = (data.get("user") or "").strip()
    if not user:
        return jsonify({"error": "user가 비어 있습니다."}), 400
    try:
        client = MongoClient(MONGO_URI)
        coll = client["user_feedback"]["feedbacks"]

        section = (data.get("section") or "").strip()
        score = data.get("score", None)
        scores_dict = data.get("scores") or None
        page = (data.get("page") or "").strip() or None
        total_score = data.get("total_score", None)

        saved = {}
        if section and (score is not None):
            doc = {"mode": "single", "user": user, "section": section, "score": int(score)}
            result = coll.insert_one(doc)
            saved = {"inserted_id": str(result.inserted_id), "section": section, "score": int(score)}
        elif isinstance(scores_dict, dict) and len(scores_dict) > 0:
            doc = {
                "mode": "bundle",
                "user": user,
                "page": page,
                "scores": {k: int(v) for k, v in scores_dict.items() if isinstance(v, (int, float))},
                "total_score": int(total_score) if isinstance(total_score, (int, float)) else None
            }
            result = coll.insert_one(doc)
            saved = {"inserted_id": str(result.inserted_id), "scores": doc["scores"], "total_score": doc["total_score"]}
        else:
            return jsonify({"error": "저장할 점수가 없습니다."}), 400

        return jsonify({"message": "overall 저장 완료", "saved": saved})
    except Exception as e:
        print("[ERROR][feedback/overall]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =============================
# [ADD] 통합 제출: CSV에 precision과 선택 아이템/장르, 테마 키워드 기록
#      + 선택된 곡은 user DB에 "모든 곡 정보"로 저장
# =============================
@app.post("/feedback/submit")
def feedback_submit():
    data = request.get_json(silent=True) or {}
    user = (data.get("user") or "").strip()
    if not user:
        return jsonify({"error": "user is required"}), 400
    sections = data.get("sections") or {}
    scores   = data.get("scores") or {}

    sec_content = sections.get("content") or {}
    sec_collabo = sections.get("collabo") or {}
    sec_hybrid  = sections.get("hybrid")  or {}
    sec_theme   = sections.get("theme")   or {}

    # enrich(장르 등 보강)
    items_c = _enrich_items_with_db(sec_content.get("items") or [])
    items_b = _enrich_items_with_db(sec_collabo.get("items") or [])
    items_h = _enrich_items_with_db(sec_hybrid.get("items") or [])
    items_t = _enrich_items_with_db(sec_theme.get("items") or [])

    prec_c = _precision(items_c, sec_content.get("total"))
    prec_b = _precision(items_b, sec_collabo.get("total"))
    prec_h = _precision(items_h, sec_hybrid.get("total"))
    prec_t = _precision(items_t, sec_theme.get("total"))

    titles_c, genres_c = _titles_and_genres(items_c)
    titles_b, genres_b = _titles_and_genres(items_b)
    titles_h, genres_h = _titles_and_genres(items_h)
    titles_t, genres_t = _titles_and_genres(items_t)
    theme_kw = _norm_str(sec_theme.get("keywords"))

    def _as_int_or_none(x):
        try:
            return int(x)
        except:
            return None
    sc_content = _as_int_or_none(scores.get("content"))
    sc_collabo = _as_int_or_none(scores.get("collabo"))
    sc_hybrid  = _as_int_or_none(scores.get("hybrid"))
    sc_theme   = _as_int_or_none(scores.get("theme"))

    # CSV append
    _ensure_csv_header()
    with open(CSV_PATH, "a", newline="", encoding="cp949") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "user": user,
            "content": f"{prec_c:.3f}",
            "collabo": f"{prec_b:.3f}",
            "hybrid":  f"{prec_h:.3f}",
            "theme":   f"{prec_t:.3f}",
            "score_content": sc_content if sc_content is not None else "",
            "score_collabo": sc_collabo if sc_collabo is not None else "",
            "score_hybrid":  sc_hybrid  if sc_hybrid  is not None else "",
            "score_theme":   sc_theme   if sc_theme   is not None else "",
            "items_content":  "; ".join(titles_c),
            "genres_content": "; ".join(genres_c),
            "items_collabo":  "; ".join(titles_b),
            "genres_collabo": "; ".join(genres_b),
            "items_hybrid":   "; ".join(titles_h),
            "genres_hybrid":  "; ".join(genres_h),
            "items_theme":    "; ".join(titles_t),
            "genres_theme":   "; ".join(genres_t),
            "theme_keywords": theme_kw
        })

    return jsonify({
        "message": "통합 제출 완료",
        "precisions": {
            "content": round(prec_c, 3),
            "collabo": round(prec_b, 3),
            "hybrid":  round(prec_h, 3),
            "theme":   round(prec_t, 3)
        },
        "scores": {   # 응답에도 포함
            "content": sc_content,
            "collabo": sc_collabo,
            "hybrid":  sc_hybrid,
            "theme":   sc_theme
        },
        "csv_path": CSV_PATH
    })

# =============================
# 엔트리포인트
# =============================
if __name__ == "__main__":
    init_all()
    app.run(debug=True)
