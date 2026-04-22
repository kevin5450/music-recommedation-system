import re

def extract_year_from_date(date_val):
    if date_val is None:
        return None
    if isinstance(date_val, int):
        return date_val if 1900 <= date_val <= 2030 else None
    # str 처리
    if isinstance(date_val, str):
        s = date_val.strip()
        if s.isdigit() and len(s) == 4:
            y = int(s)
            return y if 1900 <= y <= 2030 else None
        m = re.match(r"^\s*(\d{4})[\-\/\.\s]?", s)
        if m:
            y = int(m.group(1))
            return y if 1900 <= y <= 2030 else None
    return None


def load_user_likes(mongo_uri="mongodb://localhost:27017/"):
    from pymongo import MongoClient
    client = MongoClient(mongo_uri)
    db = client["user"]
    out = {}
    for coll_name in db.list_collection_names():
        likes = [d.get("title") for d in db[coll_name].find({}, {"_id": 0, "title": 1})]
        out[coll_name] = {t for t in likes if isinstance(t, str)}
    return list(out.keys()), out

def recommend_user_cf_overlap_proportional(
    target_user,
    user_likes_map,
    top_n=5,
    music_data=None,
    start_year=None,
    end_year=None,
):
    # --- 0) 방어 ---
    if not user_likes_map or target_user not in user_likes_map:
        return []

    # likes가 list/tuple여도 안전하게 set으로 변환
    def as_set(x):
        return set([t for t in x if isinstance(t, str)])

    target_likes = as_set(user_likes_map[target_user])

    # ---------- 1) music_data 인덱싱 ----------
    title_to_year = {}
    title_to_meta = {}
    if music_data:
        for s in music_data:
            if not isinstance(s, dict):
                continue
            t = s.get("title")
            if not isinstance(t, str) or not t.strip():
                continue
            # release_year 가 int이거나 release_date 가 str이어도 안전
            y = extract_year_from_date(s.get("release_year", None) or s.get("release_date", None))
            if y is not None:
                title_to_year[t] = y
            title_to_meta[t] = {
                "title": s.get("title", t),
                "artist": s.get("artist", "unknown"),
                "duration": s.get("duration", "--"),
                "youtube_url": s.get("youtube_url", "")
            }

    def valid_year(title):
        # 연도 필터 미사용이면 True
        if start_year is None and end_year is None:
            return True
        y = title_to_year.get(title)
        if y is None:
            # 연도 모르면 통과 (필요시 False로 되돌리면 됨)
            return True
        if start_year is not None and y < start_year:
            return False
        if end_year is not None and y > end_year:
            return False
        return True


    # ---------- 2) 이웃 겹침 계산 ----------
    overlap = {}
    for u, likes in user_likes_map.items():
        if u == target_user:
            continue
        c = len(target_likes & as_set(likes))  # ← 항상 set 교집합
        if c > 0:
            overlap[u] = c

    if not overlap:
        return []

    # ---------- 3) 비례 슬롯 배분 ----------
    total = sum(overlap.values())
    raw = {u: top_n * (c / total) for u, c in overlap.items()}
    slots = {u: int(v) for u, v in raw.items()}
    remain = top_n - sum(slots.values())
    if remain > 0:
        # 소수점 큰 순서대로 남은 슬롯 배분
        for u, _frac in sorted(((u, raw[u] - slots[u]) for u in overlap),
                               key=lambda x: x[1], reverse=True):
            if remain <= 0:
                break
            slots[u] += 1
            remain -= 1

    # ---------- 4) 후보 생성 ----------
    neighbor_candidates = {}
    for u in slots:
        cand_raw = as_set(user_likes_map[u]) - target_likes
        cand = [t for t in sorted(cand_raw) if valid_year(t)]
        neighbor_candidates[u] = cand

    # ---------- 5) 라운드로빈 픽 ----------
    picked = []
    seen = set()
    # 슬롯 있는 사용자 목록 고정 순회
    order = list(slots.keys())
    while len(picked) < top_n and any(slots[u] > 0 and neighbor_candidates[u] for u in order):
        for u in order:
            if len(picked) >= top_n:
                break
            if slots[u] <= 0:
                continue
            # 중복 제거된 첫 후보 선택
            while neighbor_candidates[u] and neighbor_candidates[u][0] in seen:
                neighbor_candidates[u].pop(0)
            if not neighbor_candidates[u]:
                continue
            title = neighbor_candidates[u].pop(0)
            if title in seen:
                continue
            meta = title_to_meta.get(title, {
                "title": title,
                "artist": "unknown",
                "duration": "--",
                "youtube_url": ""
            })
            picked.append(meta)
            seen.add(title)
            slots[u] -= 1

    # ---------- 6) 남은 슬롯 채우기 ----------
    if len(picked) < top_n:
        tail_pool = []
        for cands in neighbor_candidates.values():
            tail_pool.extend([t for t in cands if t not in seen])
        for title in tail_pool:
            if len(picked) >= top_n:
                break
            if not valid_year(title):
                continue
            meta = title_to_meta.get(title, {
                "title": title,
                "artist": "unknown",
                "duration": "--",
                "youtube_url": ""
            })
            picked.append(meta)
            seen.add(title)

    return picked[:top_n]
