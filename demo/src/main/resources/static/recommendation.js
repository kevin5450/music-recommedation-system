const API_BASE = "http://127.0.0.1:5000";
function makeRangeQuery(ranges) {
  if (!ranges || !ranges.length) return "";
  return ranges.map(([a,b]) => `&start_year=${a}&end_year=${b}`).join("");
}
const ENDPOINTS = {
  content: (user, ranges) =>
    `${API_BASE}/recommend/content?user=${encodeURIComponent(user)}${makeRangeQuery(ranges)}`,
  collab:  (user, ranges) =>
    `${API_BASE}/recommend/collab?user=${encodeURIComponent(user)}${makeRangeQuery(ranges)}`,
  hybrid:  (user, ranges) =>
    `${API_BASE}/recommend/hybrid?user=${encodeURIComponent(user)}${makeRangeQuery(ranges)}`,
  theme:   (user, q, ranges) =>
    `${API_BASE}/recommend/theme?user=${encodeURIComponent(user)}&query=${encodeURIComponent(q)}${makeRangeQuery(ranges)}`,
  feedback: `${API_BASE}/feedback/like`,
  overall:  `${API_BASE}/feedback/overall`,
  submit:   `${API_BASE}/feedback/submit`
};

/* ============== UI utils ============== */
const UI = {
  showSection(id) {
    document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');
  },
  getUser(inputId = "user-name") {
    const v = document.getElementById(inputId)?.value?.trim() || "";
    if (!v) { alert("사용자명을 입력해주세요."); throw new Error("사용자 없음"); }
    return v;
  },
  getQuery() {
    const v = document.getElementById("query")?.value?.trim() || "";
    if (!v) { alert("문장을 입력해주세요."); throw new Error("쿼리 없음"); }
    return v;
  },
  createOverallRating(section) {
    const wrap = document.createElement("div");
    wrap.className = "overall-rating";
    wrap.innerHTML = `<p style="margin:10px 0 5px;">이 추천 결과에 대한 전체적인 분위기 평가</p>`;
    const group = document.createElement("div");
    group.className = "rating-buttons";
    group.dataset.type = section;
    group.innerHTML = Array.from({length:10}, (_,i)=>`<button data-score="${i+1}">${i+1}</button>`).join("");
    wrap.appendChild(group);
    group.querySelectorAll("button").forEach(btn=>{
      btn.addEventListener("click", ()=>{
        group.querySelectorAll("button").forEach(b=>b.classList.remove("selected"));
        btn.classList.add("selected");
        console.log(`[평가 - ${section}] 전체 점수: ${btn.dataset.score}`);
      });
    });
    return wrap;
  },
  createProgress() {
    const d = document.createElement("div");
    d.className = "rating-progress";
    d.textContent = "선택한 노래: 0 / 0";
    return d;
  },
  createSubmitButton(label="전체 제출") {
    const b = document.createElement("button");
    b.className = "submit-button";
    b.textContent = label;
    b.disabled = true;
    b.title = "모든 항목을 평가해야 제출할 수 있습니다.";
    return b;
  },
  setSubmitEnabled(btn, enabled) {
    btn.disabled = !enabled;
    btn.classList.toggle("enabled", !!enabled);
    btn.title = enabled ? "" : "모든 항목을 평가해야 제출할 수 있습니다.";
  }
};

/* ============== Rating logic ============== */
function mapStarsToScore(count) {
  return (count===3) ? 3 : (count===2 ? 2 : (count===1 ? 0 : -1));
}
function attachStarEvents(starContainer, title, liEl) {
  let current = parseInt(liEl.dataset.ratingStars || "");
  const stars = starContainer.querySelectorAll('span[data-value]');
  const applyVisual = (n) => stars.forEach((s,idx)=> s.classList.toggle('selected', idx < n));
  if (!isNaN(current)) applyVisual(current);
  stars.forEach(star=>{
    const v = parseInt(star.dataset.value, 10);
    star.addEventListener('mouseover', ()=> {
      stars.forEach(s => s.classList.toggle('hovered', parseInt(s.dataset.value,10) <= v));
    });
    star.addEventListener('mouseout', ()=> {
      stars.forEach(s => s.classList.remove('hovered'));
    });
    star.addEventListener('click', ()=> {
      current = (current === v) ? 0 : v;
      applyVisual(current);
      const score = mapStarsToScore(current);
      liEl.dataset.ratingStars = String(current);
      liEl.dataset.ratingScore = String(score);
      liEl.dispatchEvent(new CustomEvent('ratingchange', { bubbles: true }));
    });
  });
}

/* ====== 선택 버튼 전용: 선택(=3★) ↔ 해제(=0★) ====== */
function attachSelectEvents(selectBtn, liEl) {
  const setState = (selected) => {
    if (selected) {
      liEl.dataset.ratingStars = "3";
      liEl.dataset.ratingScore = "3";
      selectBtn.textContent = "선택됨";
      selectBtn.classList.add("selected");
    } else {
      liEl.dataset.ratingStars = "0";
      liEl.dataset.ratingScore = "0";
      selectBtn.textContent = "선택";
      selectBtn.classList.remove("selected");
    }
    liEl.dispatchEvent(new CustomEvent('ratingchange', { bubbles: true }));
  };
  selectBtn.addEventListener("click", () => {
    const cur = liEl.dataset.ratingStars;
    const isSelected = (cur === "3");
    setState(!isSelected);
  });
}

/* ============== Item renderer ============== */
function createListItem(item) {
  const title      = item.title  || "";
  const artist     = item.artist || "";
  const duration   = item.duration || "--";
  const youtubeUrl = item.youtube_url || "";
  const genreRaw   = item.genre ?? [];
  const genreStr   = Array.isArray(genreRaw) ? genreRaw.join(", ") : String(genreRaw || "");

  const li = document.createElement("li");
  li.dataset.title       = title;
  li.dataset.artist      = artist;
  li.dataset.duration    = duration;
  li.dataset.genre       = genreStr;
  li.dataset.youtube     = youtubeUrl || "";
  // 기본값을 0으로 세팅 → 렌더 직후 평가완료 취급
  li.dataset.ratingStars = "0";
  li.dataset.ratingScore = "0";

  const yt = youtubeUrl
    ? `<a href="${youtubeUrl}" target="_blank" title="유튜브로 듣기" style="text-decoration:none;">
         <img src="https://upload.wikimedia.org/wikipedia/commons/7/75/YouTube_social_white_squircle_%282017%29.svg"
              alt="YouTube" width="24" height="24" style="vertical-align:middle;">
       </a>`
    : "";

  li.innerHTML = `
  <div class="item-row">
    <div class="item-meta">
      <div class="item-title"><strong>${title}</strong> | ${artist}</div>
      <div class="item-sub" style="margin-top:4px; display:flex; align-items:center; gap:8px;">
        <span>재생시간: ${duration}</span>
        ${yt}
      </div>
    </div>
    <div class="item-actions">
      <button type="button" class="select-btn">선택</button>
    </div>
  </div>
  `;

  const selectBtn = li.querySelector(".select-btn");
  attachSelectEvents(selectBtn, li);
  return li;
}

/* ============== Submit gating(기존 유지) ============== */
function allRated(listId) {
  const ul = document.getElementById(listId);
  if (!ul) return { ok:false, rated:0, total:0, first:null };
  const lis = Array.from(ul.querySelectorAll("li"));
  const ratedLis = lis.filter(li => li.dataset.ratingStars !== "");
  const firstUnrated = lis.find(li => li.dataset.ratingStars === "") || null;
  return { ok: ratedLis.length === lis.length, rated: ratedLis.length, total: lis.length, first: firstUnrated };
}
function updateSubmitState(listId, submitBtn, progressEl) {
  const st = allRated(listId);
  if (submitBtn && submitBtn.classList) {
    UI.setSubmitEnabled(submitBtn, st.ok);
  }
}
function collectThreeStarItems(listId) {
  const ul = document.getElementById(listId);
  if (!ul) return [];
  const rows = [];
  ul.querySelectorAll("li").forEach(li=>{
    const isThree = (li.dataset.ratingScore === "3") || (li.dataset.ratingStars === "3");
    if (isThree) {
      rows.push({
        title:    li.dataset.title  || "",
        artist:   li.dataset.artist || "",
        duration: li.dataset.duration || ""
      });
    }
  });
  return rows;
}

/* ============== [ADD] 통합 제출용 유틸 ============== */
function collectSelectedItems(listId) {
  const ul = document.getElementById(listId);
  if (!ul) return [];
  const rows = [];
  ul.querySelectorAll("li").forEach(li=>{
    const isThree = (li.dataset.ratingScore === "3") || (li.dataset.ratingStars === "3");
    if (isThree) {
      rows.push({
        title:       li.dataset.title  || "",
        artist:      li.dataset.artist || "",
        genre:       li.dataset.genre  || "",
        youtube_url: li.dataset.youtube || ""
      });
    }
  });
  return rows;
}
function countItems(listId) {
  const ul = document.getElementById(listId);
  return ul ? ul.querySelectorAll("li").length : 0;
}
async function sendSubmitCombined(payload) {
  const res = await fetch(ENDPOINTS.submit, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  return res.json();
}

async function sendOverallToServerSingle({ user, section, score }) {
  const res = await fetch(ENDPOINTS.overall, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ user, section, score })
  });
  return res.json();
}

/* ============== Year buttons (multi-select, no auto-fetch) ============== */
const YearBar = {
  selectedTokensByBox: Object.create(null),

  parseRange(str) {
    if (!str) return null;
    const m = str.match(/^(\d{4})(?:-(\d{4})?)?$/); // "2021-" 허용
    if (!m) return null;
    const a = parseInt(m[1], 10);
    const b = m[2] ? parseInt(m[2], 10) : 2025;
    if (Number.isNaN(a) || Number.isNaN(b)) return null;
    return [a, b];
  },

  getRanges(scopeKey) {
    const tokens = this.selectedTokensByBox[scopeKey] || new Set();
    const out = [];
    tokens.forEach(tok => {
      const r = this.parseRange(tok);
      if (r) out.push(r);
    });
    return out; // Array<[start,end]>
  },

  bind(containerId, scopeKey) {
    const wrap = document.getElementById(containerId);
    if (!wrap) return;
    if (!this.selectedTokensByBox[scopeKey]) this.selectedTokensByBox[scopeKey] = new Set();

    wrap.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-range]");
      if (!btn) return;

      const tok = String(btn.dataset.range || "");
      const set = YearBar.selectedTokensByBox[scopeKey];

      if (btn.classList.contains("selected")) {
        btn.classList.remove("selected");
        set.delete(tok);
      } else {
        btn.classList.add("selected");
        set.add(tok);
      }
      // 호출은 버튼(모든 추천 받기)에서만
    });
  }
};

/* ============== 보조: 섹션 오버롤 점수 가져오기 ============== */
function getOverallScore(section) {
  const group = document.querySelector(`.rating-buttons[data-type="${section}"]`);
  const sel = group?.querySelector("button.selected");
  return sel ? parseInt(sel.dataset.score, 10) : -1;
}

/* ============== 미평가 0으로 정규화 ============== */
function normalizeUnrated(listId) {
  const ul = document.getElementById(listId);
  if (!ul) return;
  ul.querySelectorAll("li").forEach(li => {
    if (li.dataset.ratingStars === "" || li.dataset.ratingScore === "") {
      li.dataset.ratingStars = "0";
      li.dataset.ratingScore = "0";
      li.dispatchEvent(new CustomEvent('ratingchange', { bubbles: true }));
    }
  });
}

/* ============== fetch/merge helpers (단일 호출) ============== */
async function fetchAndRender(url, listId, boxType) {
  const ul = document.getElementById(listId);
  const box = ul.closest(".recommend-box");

  // submit-button은 지우지 않는다(재사용)
  box.querySelectorAll(".overall-rating, .rating-progress").forEach(n => n.remove());
  ul.innerHTML = `<li>불러오는 중…</li>`;

  try {
    const r = await fetch(url);
    const data = await r.json();
    const recommendations = Array.isArray(data?.recommendations) ? data.recommendations : [];

    if (!recommendations.length) {
      ul.innerHTML = `<li>추천 결과가 없습니다.</li>`;
      if (boxType === "theme") {
        // 테마 박스에 버튼이 없으면 만들어 두기(비활성)
        let submitBtn = box.querySelector(".submit-button");
        if (!submitBtn) {
          submitBtn = UI.createSubmitButton("전체 제출");
          box.appendChild(submitBtn);
        }
      }
      return;
    }

    ul.innerHTML = "";
    recommendations.forEach(it => ul.appendChild(createListItem(it)));

    const ratingPanel = UI.createOverallRating(boxType);
    box.appendChild(ratingPanel);

    const progress = UI.createProgress();
    box.appendChild(progress);

    if (boxType === "theme") {
      // 버튼 재사용(없으면 생성)
      let submitBtn = box.querySelector(".submit-button");
      if (!submitBtn) {
        submitBtn = UI.createSubmitButton("전체 제출");
        box.appendChild(submitBtn);
      }
      // 중복 바인딩 방지
      submitBtn.onclick = async () => {
        // 안전망: 모든 리스트 미평가 0으로 정규화
        normalizeUnrated("cf-results");
        normalizeUnrated("collab-results");
        normalizeUnrated("hybrid-results");
        normalizeUnrated("theme-result-list");

        // 네 섹션이 모두 로드/평가되었는지 확인
        const lists = [
          { id: "cf-results",        key: "content", label: "콘텐츠" },
          { id: "collab-results",    key: "collabo", label: "협업"   },
          { id: "hybrid-results",    key: "hybrid",  label: "하이브리드" },
          { id: "theme-result-list", key: "theme",   label: "테마"   },
        ];
        for (const L of lists) {
          if (countItems(L.id) === 0) {
            alert(`[${L.label}] 추천을 먼저 불러오세요.`);
            return;
          }
          const st = allRated(L.id);
          if (!st.ok) {
            alert(`[${L.label}] 아직 평가하지 않은 항목이 있습니다. (${st.rated}/${st.total})`);
            if (st.first) st.first.scrollIntoView({ behavior:"smooth", block:"center" });
            return;
          }
        }

        try {
          const user = UI.getUser("user-name-theme");
          const q    = UI.getQuery();
          const sections = {
            content: { items: collectSelectedItems("cf-results"),        total: countItems("cf-results") },
            collabo: { items: collectSelectedItems("collab-results"),    total: countItems("collab-results") },
            hybrid:  { items: collectSelectedItems("hybrid-results"),    total: countItems("hybrid-results") },
            theme:   { items: collectSelectedItems("theme-result-list"), total: countItems("theme-result-list"), keywords: q }
          };
          const scores = {
            content: getOverallScore("content"),
            collabo: getOverallScore("collab"),
            hybrid:  getOverallScore("hybrid"),
            theme:   getOverallScore("theme"),
          };
          const out = await sendSubmitCombined({ user, sections, scores });
          alert(out.message || "제출 완료");
        } catch (e) {
          console.error(e);
          alert("통합 제출 중 오류가 발생했습니다.");
        }
      };

      // 테마 진행도로 버튼 활성화(실제 검증은 클릭 시 일괄)
      normalizeUnrated("theme-result-list");
      box.addEventListener("ratingchange", () => {
        updateSubmitState(listId, box.querySelector(".submit-button"), progress);
      });
      updateSubmitState(listId, box.querySelector(".submit-button"), progress);

    } else {
      // 다른 섹션: 제출 버튼 없음, 진행도만 표시 갱신
      box.addEventListener("ratingchange", () => {
        updateSubmitState(listId, { disabled: true, classList:{toggle:()=>{}}, title:"" }, progress);
      });
      updateSubmitState(listId, { disabled: true, classList:{toggle:()=>{}}, title:"" }, progress);
    }

  } catch (e) {
    console.error(e);
    ul.innerHTML = `<li>오류가 발생했습니다.</li>`;
  }
}

/* ============== Endpoint builders (단일 URL) ============== */
function buildEndpointsPersonal(user, ranges) {
  return {
    content: ENDPOINTS.content(user, ranges),
    collab:  ENDPOINTS.collab(user,  ranges),
    hybrid:  ENDPOINTS.hybrid(user,  ranges),
  };
}
function buildEndpointsTheme(user, q, ranges) {
  return ENDPOINTS.theme(user, q, ranges);
}

/* ============== Actions (triggered ONLY by buttons) ============== */
async function getAllRecommendations() {
  const user = UI.getUser("user-name");
  const ranges = YearBar.getRanges("personal");
  const eps = buildEndpointsPersonal(user, ranges);
  await Promise.all([
    fetchAndRender(eps.content, "cf-results",    "content"),
    fetchAndRender(eps.collab,  "collab-results","collab"),
    fetchAndRender(eps.hybrid,  "hybrid-results","hybrid"),
  ]);
}
async function getTheme() {
  const user = UI.getUser("user-name-theme");
  const q = UI.getQuery();
  const ranges = YearBar.getRanges("theme");
  const ep = buildEndpointsTheme(user, q, ranges);
  await fetchAndRender(ep, "theme-result-list", "theme");
}

/* ============== Bind ============== */
document.getElementById("btn-personal-all")?.addEventListener("click", getAllRecommendations);
document.getElementById("btn-theme-all")?.addEventListener("click", getTheme);
// 과거 하단 제출 버튼은 HTML에 없음(동적 버튼만 사용)

YearBar.bind("year-bar-personal", "personal");
YearBar.bind("year-bar-theme", "theme");
