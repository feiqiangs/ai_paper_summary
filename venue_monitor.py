"""
venue_monitor.py
~~~~~~~~~~~~~~~~
从 DBLP 抓取顶会论文（OSDI / SOSP / NSDI / MLSys / ASPLOS / EuroSys /
SIGMOD / VLDB / SIGCOMM），与 paper_monitor.py 中的关键词做标题匹配，
结果写入 SQLite 缓存，并提供 HTML 片段供报告使用。

依赖：requests（已有）、sqlite3（标准库）
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / ".." / "paper_monitor_cache.db"

# 顶会列表：(展示名, DBLP stream key)
VENUES: list[tuple[str, str]] = [
    ("OSDI",    "conf/osdi"),
    ("SOSP",    "conf/sosp"),
    ("NSDI",    "conf/nsdi"),
    ("MLSys",   "conf/mlsys"),
    ("ASPLOS",  "conf/asplos"),
    ("EuroSys", "conf/eurosys"),
    ("SIGMOD",  "conf/sigmod"),
    ("VLDB",    "journals/pvldb"),
    ("SIGCOMM", "conf/sigcomm"),
]

# 缓存有效期（秒），每天刷新一次
CACHE_TTL = 24 * 3600

# DBLP API
DBLP_API = "https://dblp.org/search/publ/api"

# 每个会议单次最大拉取量（DBLP 上限 200）
MAX_PER_VENUE = 200

# 动态词典数据库路径（复用 github_monitor_cache.db）
DOMAIN_DB_PATH = Path(__file__).parent / ".." / "github_monitor_cache.db"

# 内置 fallback 关键词（当动态词典为空时使用）
_FALLBACK_KEYWORDS: list[tuple[str, str]] = [
    ("llm inference",              "LLM Inference"),
    ("inference optimization",     "Inference Optimization"),
    ("serving system",             "Serving System"),
    ("disaggregated inference",    "Disaggregated Inference"),
    ("inference framework",        "Inference Framework"),
    ("llm serving",                "LLM Serving"),
    ("vllm",                       "vLLM"),
    ("sglang",                     "SGLang"),
    ("tensorrt-llm",               "TensorRT-LLM"),
    ("tensorrt",                   "TensorRT"),
    ("deepseek",                   "DeepSeek"),
    ("qwen",                       "Qwen"),
    ("kv cache",                   "KV Cache"),
    ("kv-cache",                   "KV-Cache"),
    ("kvcache",                    "KVCache"),
    ("attention cache",            "Attention Cache"),
    ("prefix cache",               "Prefix Cache"),
    ("key-value cache",            "Key-Value Cache"),
    ("speculative decoding",       "Speculative Decoding"),
    ("quantization",               "Quantization"),
    ("model compression",          "Model Compression"),
    ("flash attention",            "Flash Attention"),
    ("paged attention",            "Paged Attention"),
    ("continuous batching",        "Continuous Batching"),
    ("model parallelism",          "Model Parallelism"),
    ("tensor parallelism",         "Tensor Parallelism"),
    ("pipeline parallelism",       "Pipeline Parallelism"),
    ("distributed inference",      "Distributed Inference"),
    ("mixture of experts",         "Mixture of Experts"),
    ("moe",                        "MoE"),
    ("prefill",                    "Prefill"),
    ("gpu scheduling",             "GPU Scheduling"),
    ("memory management",          "Memory Management"),
    ("memory efficiency",          "Memory Efficiency"),
    ("long context",               "Long Context"),
    ("large language model",       "Large Language Model"),
    ("transformer",                "Transformer"),
    ("autoregressive",             "Autoregressive"),
    ("model serving",              "Model Serving"),
    ("disaggregated serving",      "Disaggregated Serving"),
]

# 动态词典缓存（懒加载）
_dynamic_kw_cache: list[tuple[str, str]] | None = None


def _load_domain_keywords() -> list[tuple[str, str]]:
    """
    从 github_monitor_cache.db 的 domain_phrases 表加载 AI Infra 领域词典。
    返回 [(phrase_lower, display_name)] 按短语长度降序（长短语优先匹配）。
    合并动态词典 + 内置 fallback，去重后返回。
    """
    global _dynamic_kw_cache
    if _dynamic_kw_cache is not None:
        return _dynamic_kw_cache

    # 噪音词过滤：这些词来自代码/PR/通用上下文，不是 AI Infra 领域词
    _NOISE_WORDS = {
        "fast", "core", "warp", "cool", "dream", "solar", "muse", "prism",
        "mirage", "capt", "ctrl", "casr", "mebm", "jepa", "chammi",
        "readme", "bugfix", "efficient and", "based optimization",
        "model already", "shared use", "model tests", "model submission",
        "pipeline ensures", "free streaming", "pay attention",
        "how alignment", "the alignment", "without training",
        "multiple alignment", "matching alignment",
        "do_lower_case", "distributed_test", "manager_tests",
        "autotokenizer", "subgraphtracer",
    }

    dynamic: list[tuple[str, str]] = []
    if DOMAIN_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DOMAIN_DB_PATH.resolve()))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT phrase_lower, display_name FROM domain_phrases "
                "WHERE freq >= 2 ORDER BY LENGTH(phrase_lower) DESC"
            ).fetchall()
            conn.close()
            # 过滤噪音词
            dynamic = [
                (r["phrase_lower"], r["display_name"])
                for r in rows
                if r["phrase_lower"] not in _NOISE_WORDS
                and not r["phrase_lower"].startswith("_")  # 过滤 Python 内部符号
                and len(r["phrase_lower"]) >= 3            # 过滤过短词
            ]
            print(f"[venue_monitor] 动态词典已加载：{len(dynamic)} 条（过滤噪音后）")
        except Exception as e:
            print(f"[venue_monitor] 动态词典加载失败: {e}")

    # 合并：动态词典优先，fallback 补充（避免重复）
    seen = {p for p, _ in dynamic}
    combined = list(dynamic)
    for phrase_lower, display in _FALLBACK_KEYWORDS:
        if phrase_lower not in seen:
            combined.append((phrase_lower, display))
            seen.add(phrase_lower)

    # 按短语长度降序（长短语优先，避免 "kv" 先于 "kv cache" 命中）
    combined.sort(key=lambda x: len(x[0]), reverse=True)
    _dynamic_kw_cache = combined
    return _dynamic_kw_cache


def _match_keywords(title: str) -> list[str]:
    """
    对论文标题做精确关键词匹配（不区分大小写）。
    使用 domain_phrases SQLite 词典 + 内置 fallback 关键词。
    返回命中的 display_name 列表（按短语长度降序，最多 8 个）。
    """
    title_lower = title.lower()
    keywords = _load_domain_keywords()
    matched: list[str] = []
    matched_positions: list[tuple[int, int]] = []  # 记录已匹配区间，避免子串重复

    for phrase_lower, display in keywords:
        idx = title_lower.find(phrase_lower)
        if idx == -1:
            continue
        end = idx + len(phrase_lower)
        # 检查是否和已匹配区间重叠（长短语优先已保证，这里防止短语被重复计入）
        overlap = any(s <= idx < e or s < end <= e or (idx <= s and end >= e)
                      for s, e in matched_positions)
        if not overlap:
            matched.append(display)
            matched_positions.append((idx, end))
        if len(matched) >= 8:
            break

    return matched


# ──────────────────────────────────────────────────────────────
# SQLite 初始化
# ──────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH.resolve()))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""
        CREATE TABLE IF NOT EXISTS venue_papers (
            dblp_key     TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            venue        TEXT NOT NULL,
            year         INTEGER,
            authors      TEXT,
            dblp_url     TEXT,
            doi_url      TEXT,
            matched_kws  TEXT,
            fetched_at   TEXT NOT NULL,
            confirmed    INTEGER DEFAULT 1,
            hotness_score REAL DEFAULT 0.0
        )
    """)
    # 兼容旧版本：若 hotness_score 列不存在则添加
    cols = [r[1] for r in db.execute("PRAGMA table_info(venue_papers)").fetchall()]
    if "hotness_score" not in cols:
        db.execute("ALTER TABLE venue_papers ADD COLUMN hotness_score REAL DEFAULT 0.0")
    db.execute("""
        CREATE TABLE IF NOT EXISTS venue_fetch_log (
            venue        TEXT NOT NULL,
            year         INTEGER NOT NULL,
            total_papers INTEGER,
            matched      INTEGER,
            fetched_at   TEXT NOT NULL,
            PRIMARY KEY (venue, year)
        )
    """)
    db.commit()
    return db


# ──────────────────────────────────────────────────────────────
# 顶会论文热度评分
# ──────────────────────────────────────────────────────────────

def compute_venue_hotness(
    title: str,
    matched_kws: list[str],
    year: int,
    venue: str,
    *,
    cite_count: int = 0,
    github_stars: int = 0,
    has_code: bool = False,
) -> float:
    """
    计算顶会论文热度分（满分 100 分），适配无 arXiv/HF 数据的场景。

    评分维度：
    - 引用次数               40分：log 归一化（200引用约满分，引用是最强信号）
    - 时效性                 20分：当年论文 20 分，去年 10 分，前年 4 分
    - 会议权重               15分：顶会分级加权
    - 关键词命中质量         15分：高价值词 2 分/个，普通词 0.5 分/个
    - GitHub / 代码收录      10分：有代码 +5，star 数线性加分
    """
    import math

    score = 0.0

    # ── 1. 关键词命中质量（15分）──
    # 高价值词（精确技术方向）加权 2 分，普通词 0.5 分，上限 15 分
    HIGH_VALUE_KW = {
        "kv cache", "kv-cache", "kvcache", "speculative decoding",
        "disaggregated inference", "disaggregated serving",
        "flash attention", "paged attention", "continuous batching",
        "vllm", "sglang", "tensorrt-llm", "deepseek", "prefix cache",
        "mixture of experts", "model parallelism", "tensor parallelism",
    }
    kw_score = 0.0
    for kw in matched_kws:
        if kw.lower() in HIGH_VALUE_KW:
            kw_score += 2.0
        else:
            kw_score += 0.5
    score += min(kw_score, 15.0)

    # ── 2. 引用次数（40分）──
    # log 归一化：log2(cite+1)/log2(201)*40，200引用约得满分
    if cite_count > 0:
        cite_score = math.log2(cite_count + 1) / math.log2(201) * 40.0
        score += min(cite_score, 40.0)

    # ── 3. 时效性（20分）──
    cur_year = datetime.now().year
    age = cur_year - year
    if age == 0:
        score += 20.0
    elif age == 1:
        score += 10.0
    elif age == 2:
        score += 4.0
    # 更早不加分

    # ── 4. 会议权重（15分）──
    VENUE_TIER = {
        "OSDI": 15, "SOSP": 15,           # 系统顶会
        "NSDI": 13, "SIGCOMM": 13,        # 网络顶会
        "MLSys": 14,                       # ML系统专属
        "ASPLOS": 12, "EuroSys": 11,      # 体系结构/欧洲系统
        "SIGMOD": 10, "VLDB": 10,         # 数据库
    }
    score += VENUE_TIER.get(venue, 8)

    # ── 5. GitHub / 代码收录（10分）──
    if has_code or github_stars > 0:
        star_score = min(github_stars, 1000) / 1000.0 * 5.0
        score += min(5.0 + star_score, 10.0)

    return round(score, 2)


# ──────────────────────────────────────────────────────────────
# DBLP 抓取
# ──────────────────────────────────────────────────────────────

def _dblp_fetch_venue(stream: str, year: int, retries: int = 3) -> list[dict]:
    """拉取某会议某年全部论文，自动分页，返回原始 hit list。"""
    all_hits: list[dict] = []
    batch = 200
    offset = 0
    while True:
        for attempt in range(retries):
            try:
                resp = requests.get(
                    DBLP_API,
                    params={
                        "q": f"stream:{stream}: year:{year}",
                        "format": "json",
                        "h": batch,
                        "f": offset,
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
                hits_obj = data.get("result", {}).get("hits", {})
                total = int(hits_obj.get("@total", 0))
                hits = hits_obj.get("hit", [])
                all_hits.extend(hits)
                # 判断是否还有更多
                if offset + len(hits) >= total:
                    return all_hits
                offset += len(hits)
                time.sleep(0.4)
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"[venue_monitor] DBLP fetch error ({stream} {year}): {e}")
                    return all_hits
                time.sleep(1.5 * (attempt + 1))
    return all_hits


def _is_cache_fresh(venue: str, year: int, db: sqlite3.Connection) -> bool:
    """检查该会议+年份的缓存是否在 TTL 内。"""
    row = db.execute(
        "SELECT fetched_at FROM venue_fetch_log WHERE venue=? AND year=?",
        (venue, year)
    ).fetchone()
    if not row:
        return False
    fetched = datetime.fromisoformat(row["fetched_at"])
    return (datetime.now() - fetched).total_seconds() < CACHE_TTL


def fetch_and_cache_venue(venue_name: str, stream: str, year: int,
                          db: sqlite3.Connection, force: bool = False) -> int:
    """
    拉取某会议某年的论文并写入 SQLite。
    返回命中关键词的论文数量。
    """
    if not force and _is_cache_fresh(venue_name, year, db):
        matched = db.execute(
            "SELECT COUNT(*) as c FROM venue_papers WHERE venue=? AND year=? AND matched_kws!=''",
            (venue_name, year)
        ).fetchone()["c"]
        return matched

    hits = _dblp_fetch_venue(stream, year)
    now_str = datetime.now().isoformat(timespec="seconds")
    matched_count = 0

    for hit in hits:
        info = hit.get("info", {})
        title = info.get("title", "").strip()
        if not title:
            continue

        dblp_key = info.get("key", hit.get("@id", ""))
        authors_raw = info.get("authors", {})
        if isinstance(authors_raw, dict):
            a = authors_raw.get("author", [])
            if isinstance(a, str):
                authors = a
            elif isinstance(a, list):
                authors = ", ".join(
                    x.get("text", x) if isinstance(x, dict) else str(x)
                    for x in a[:6]
                )
            else:
                authors = ""
        else:
            authors = str(authors_raw)

        dblp_url = info.get("url", "")
        doi_url = ""
        ee = info.get("ee", "")
        if isinstance(ee, str) and ee.startswith("http"):
            doi_url = ee
        elif isinstance(ee, list):
            for e in ee:
                if isinstance(e, str) and e.startswith("http"):
                    doi_url = e
                    break

        matched_kws = _match_keywords(title)
        if matched_kws:
            matched_count += 1

        # 计算热度分（无引用/star数据时只用关键词+时效性+会议权重）
        hotness = compute_venue_hotness(
            title=title,
            matched_kws=matched_kws,
            year=int(info.get("year", year)),
            venue=venue_name,
        )

        db.execute("""
            INSERT OR REPLACE INTO venue_papers
              (dblp_key, title, venue, year, authors, dblp_url, doi_url, matched_kws, fetched_at, hotness_score)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            dblp_key, title, venue_name, int(info.get("year", year)),
            authors, dblp_url, doi_url,
            json.dumps(matched_kws, ensure_ascii=False),
            now_str,
            hotness,
        ))

    db.execute("""
        INSERT OR REPLACE INTO venue_fetch_log (venue, year, total_papers, matched, fetched_at)
        VALUES (?,?,?,?,?)
    """, (venue_name, year, len(hits), matched_count, now_str))
    db.commit()
    return matched_count


# ──────────────────────────────────────────────────────────────
# 主刷新入口
# ──────────────────────────────────────────────────────────────

def refresh_all_venues(years: Optional[list[int]] = None, force: bool = False) -> dict:
    """
    刷新所有顶会的论文缓存。
    years: 要拉取的年份列表，默认 [当前年, 当前年-1]
    返回 {venue_name: matched_count} 统计。
    """
    if years is None:
        cur_year = datetime.now().year
        years = [cur_year, cur_year - 1]

    db = _get_db()
    stats: dict[str, int] = {}

    for venue_name, stream in VENUES:
        total_matched = 0
        for year in years:
            cnt = fetch_and_cache_venue(venue_name, stream, year, db, force=force)
            total_matched += cnt
            time.sleep(0.3)
        stats[venue_name] = total_matched
        print(f"[venue_monitor] {venue_name}: {total_matched} matched papers cached")

    db.close()
    return stats


# ──────────────────────────────────────────────────────────────
# 查询匹配论文
# ──────────────────────────────────────────────────────────────

def get_matched_papers(
    years: Optional[list[int]] = None,
    limit_per_venue: int = 5,
) -> dict[str, list[dict]]:
    """
    返回各顶会命中关键词的论文，按 venue 分组。
    结构：{venue_name: [{"title", "year", "authors", "doi_url", "dblp_url", "matched_kws"}, ...]}
    """
    if years is None:
        cur_year = datetime.now().year
        years = [cur_year, cur_year - 1]

    db = _get_db()
    result: dict[str, list[dict]] = {}

    for venue_name, _ in VENUES:
        rows = db.execute("""
            SELECT title, year, authors, doi_url, dblp_url, matched_kws, hotness_score
            FROM venue_papers
            WHERE venue=? AND year IN ({placeholders}) AND matched_kws != '[]' AND matched_kws != ''
            ORDER BY hotness_score DESC, year DESC
            LIMIT ?
        """.format(placeholders=",".join("?" * len(years))),
            (venue_name, *years, limit_per_venue)
        ).fetchall()

        result[venue_name] = [
            {
                "title": r["title"],
                "year": r["year"],
                "authors": r["authors"],
                "doi_url": r["doi_url"],
                "dblp_url": r["dblp_url"],
                "matched_kws": json.loads(r["matched_kws"]) if r["matched_kws"] else [],
                "hotness_score": round(r["hotness_score"] or 0.0, 1),
            }
            for r in rows
        ]

    db.close()
    return result


# ──────────────────────────────────────────────────────────────
# HTML 片段生成
# ──────────────────────────────────────────────────────────────

def build_venue_section_html(
    years: Optional[list[int]] = None,
    limit_per_venue: int = 10,
    auto_refresh: bool = True,
) -> str:
    """
    生成顶会论文 HTML 板块，供 paper_monitor.build_html_report 插入。
    auto_refresh=True 时会先尝试刷新缓存（受 TTL 保护，每天最多请求一次）。
    """
    if years is None:
        cur_year = datetime.now().year
        years = [cur_year, cur_year - 1]

    if auto_refresh:
        try:
            refresh_all_venues(years=years, force=False)
        except Exception as e:
            print(f"[venue_monitor] refresh error: {e}")

    matched = get_matched_papers(years=years, limit_per_venue=limit_per_venue)

    # 统计
    total_matched = sum(len(v) for v in matched.values())
    active_venues = sum(1 for v in matched.values() if v)
    years_label = " / ".join(str(y) for y in sorted(years, reverse=True))

    if total_matched == 0:
        return f"""
  <p class="section-title">🎓 顶会论文追踪（{years_label}）</p>
  <div class="venue-empty">暂无命中关键词的顶会论文，请等待数据刷新</div>
"""

    # 构建每个 venue 的卡片
    venue_cards = ""
    for venue_name, papers in matched.items():
        if not papers:
            continue

        badge_color_map = {
            "OSDI": "#4f8ef7", "SOSP": "#4f8ef7",
            "NSDI": "#34d399", "MLSys": "#f97316",
            "ASPLOS": "#a78bfa", "EuroSys": "#a78bfa",
            "SIGMOD": "#facc15", "VLDB": "#facc15",
            "SIGCOMM": "#34d399",
        }
        badge_color = badge_color_map.get(venue_name, "#64748b")

        rows = ""
        for p in papers:
            title_esc = p["title"].replace("<", "&lt;").replace(">", "&gt;")
            authors_short = p["authors"]
            if authors_short and len(authors_short) > 60:
                authors_short = authors_short[:57] + "…"
            authors_esc = authors_short.replace("<", "&lt;").replace(">", "&gt;") if authors_short else ""

            # 链接优先 doi > dblp
            link_url = p["doi_url"] or p["dblp_url"] or "#"
            link_url_esc = link_url.replace('"', "&quot;")

            # 关键词 badges（最多 4 个，来自动态词典）
            kw_badges = "".join(
                f'<span class="venue-kw-badge">{kw}</span>'
                for kw in p["matched_kws"][:4]
            )

            # 热度分徽章
            hscore = p.get("hotness_score", 0.0)
            if hscore >= 70:
                heat_cls = "venue-heat-high"
            elif hscore >= 50:
                heat_cls = "venue-heat-mid"
            else:
                heat_cls = "venue-heat-low"
            heat_badge = f'<span class="venue-heat-badge {heat_cls}">{hscore:.0f}分</span>'

            rows += f"""
        <tr class="venue-paper-row">
          <td class="venue-title-cell">
            <a href="{link_url_esc}" target="_blank" class="venue-paper-link">{title_esc}</a>
          </td>
          <td class="venue-authors-cell">{authors_esc}</td>
          <td class="venue-year-cell">{p["year"]}</td>
          <td class="venue-kw-cell">{kw_badges}</td>
          <td class="venue-score-cell">{heat_badge}</td>
        </tr>"""

        venue_cards += f"""
      <div class="venue-card">
        <div class="venue-card-header">
          <span class="venue-badge" style="background:{badge_color}20;color:{badge_color};border-color:{badge_color}40">{venue_name}</span>
          <span class="venue-count">{len(papers)} 篇命中（按热度排序）</span>
        </div>
        <div class="venue-table-scroll">
        <table class="venue-table">
          <thead>
            <tr>
              <th>标题</th>
              <th>作者</th>
              <th>年份</th>
              <th>命中关键词</th>
              <th>热度分</th>
            </tr>
          </thead>
          <tbody>{rows}
          </tbody>
        </table>
        </div>
      </div>"""

    html = f"""
  <p class="section-title">🎓 顶会论文追踪 — {years_label} · {active_venues} 个会议 · {total_matched} 篇命中</p>
  <div class="venue-meta">
    数据来源 <a href="https://dblp.org" target="_blank" class="venue-src-link">DBLP</a>，
    覆盖 OSDI · SOSP · NSDI · MLSys · ASPLOS · EuroSys · SIGMOD · VLDB · SIGCOMM，
    关键词匹配基于 AI Infra 领域词典，每会议按热度最多显示 {limit_per_venue} 篇
  </div>
  <div class="venue-grid">
    {venue_cards}
  </div>
"""
    return html


# ──────────────────────────────────────────────────────────────
# CSS（注入到 paper_monitor 的 <style> 中）
# ──────────────────────────────────────────────────────────────

VENUE_CSS = """
  /* ── 顶会论文板块 ── */
  .venue-meta {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: -8px 0 20px;
    line-height: 1.6;
  }
  .venue-src-link {
    color: var(--accent);
    text-decoration: none;
  }
  .venue-src-link:hover { text-decoration: underline; }

  .venue-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
  }

  .venue-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }

  .venue-card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
  }

  .venue-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid;
    letter-spacing: 0.05em;
  }

  .venue-count {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  /* 表格横向滚动容器 */
  .venue-table-scroll {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  .venue-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    min-width: 480px;
  }

  .venue-table thead tr {
    background: rgba(255,255,255,0.02);
  }

  .venue-table th {
    padding: 8px 12px;
    text-align: left;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
  }

  .venue-paper-row:hover { background: rgba(255,255,255,0.03); }

  .venue-paper-row td {
    padding: 8px 12px;
    vertical-align: top;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }

  .venue-paper-row:last-child td { border-bottom: none; }

  .venue-title-cell { max-width: 320px; }

  .venue-paper-link {
    color: var(--text);
    text-decoration: none;
    line-height: 1.4;
    display: block;
  }
  .venue-paper-link:hover { color: var(--accent); }

  .venue-authors-cell {
    color: var(--text-muted);
    font-size: 0.72rem;
    max-width: 160px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .venue-year-cell {
    color: var(--text-dim);
    font-size: 0.72rem;
    white-space: nowrap;
  }

  .venue-kw-cell { max-width: 180px; }

  .venue-kw-badge {
    display: inline-block;
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 4px;
    background: rgba(79,142,247,0.12);
    color: #7eb3ff;
    border: 1px solid rgba(79,142,247,0.2);
    margin: 2px 2px 2px 0;
    white-space: nowrap;
  }

  .venue-score-cell {
    white-space: nowrap;
    text-align: center;
  }

  .venue-heat-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 5px;
    white-space: nowrap;
    font-family: 'Space Mono', monospace;
  }

  .venue-heat-high {
    background: rgba(249,115,22,0.15);
    color: #fb923c;
    border: 1px solid rgba(249,115,22,0.3);
  }

  .venue-heat-mid {
    background: rgba(250,204,21,0.12);
    color: #fbbf24;
    border: 1px solid rgba(250,204,21,0.25);
  }

  .venue-heat-low {
    background: rgba(100,116,139,0.12);
    color: #94a3b8;
    border: 1px solid rgba(100,116,139,0.2);
  }

  .venue-empty {
    padding: 24px;
    color: var(--text-muted);
    font-size: 0.85rem;
    text-align: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 32px;
  }

  @media (max-width: 768px) {
    .venue-grid { grid-template-columns: 1fr; }
    .venue-authors-cell { display: none; }
    .venue-table th:nth-child(2) { display: none; }
    .venue-title-cell { max-width: 200px; }
    .venue-kw-cell { max-width: 120px; }
  }

  @media (max-width: 480px) {
    .venue-table { font-size: 0.72rem; }
    .venue-kw-badge { font-size: 0.6rem; padding: 1px 5px; }
    .venue-card-header { padding: 10px 12px; }
  }
"""


# ──────────────────────────────────────────────────────────────
# Semantic Scholar 摘要补全
# ──────────────────────────────────────────────────────────────

S2_API = "https://api.semanticscholar.org/graph/v1/paper"
_S2_HEADERS = {"User-Agent": "venue-monitor/1.0 (research tool)"}


def fetch_abstract_s2(title: str, doi_url: str = "") -> str:
    """通过 Semantic Scholar 补全摘要，优先 DOI，其次标题搜索。"""
    # 方式1：DOI 直接查询
    if doi_url and "doi.org" in doi_url:
        doi = doi_url.split("doi.org/")[-1]
        try:
            resp = requests.get(
                f"{S2_API}/DOI:{doi}",
                params={"fields": "abstract"},
                headers=_S2_HEADERS, timeout=10
            )
            if resp.status_code == 200:
                abstract = resp.json().get("abstract", "")
                if abstract:
                    return abstract.strip()
        except Exception:
            pass

    # 方式2：标题搜索
    try:
        resp = requests.get(
            f"{S2_API}/search",
            params={"query": title, "fields": "abstract,title", "limit": 3},
            headers=_S2_HEADERS, timeout=10
        )
        if resp.status_code == 200:
            for paper in resp.json().get("data", []):
                s2_title = paper.get("title", "").lower()
                if title.lower()[:30] in s2_title or s2_title[:30] in title.lower():
                    abstract = paper.get("abstract", "")
                    if abstract:
                        return abstract.strip()
    except Exception:
        pass
    return ""


def enrich_abstracts(years: Optional[list[int]] = None, limit: int = 200):
    """
    对命中关键词但尚无摘要的论文，批量从 S2 补全摘要。
    """
    if years is None:
        years = [datetime.now().year]

    db = _get_db()

    # 检查 abstract 列是否存在，不存在则添加
    cols = [r[1] for r in db.execute("PRAGMA table_info(venue_papers)").fetchall()]
    if "abstract" not in cols:
        db.execute("ALTER TABLE venue_papers ADD COLUMN abstract TEXT DEFAULT ''")
        db.commit()
    if "abstract_zh" not in cols:
        db.execute("ALTER TABLE venue_papers ADD COLUMN abstract_zh TEXT DEFAULT ''")
        db.commit()

    placeholders = ",".join("?" * len(years))
    rows = db.execute(f"""
        SELECT dblp_key, title, doi_url FROM venue_papers
        WHERE year IN ({placeholders})
          AND matched_kws != '[]' AND matched_kws != ''
          AND (abstract IS NULL OR abstract = '')
        LIMIT ?
    """, (*years, limit)).fetchall()

    print(f"[venue_monitor] 需要补全摘要: {len(rows)} 篇")
    for i, row in enumerate(rows):
        abstract = fetch_abstract_s2(row["title"], row["doi_url"] or "")
        db.execute(
            "UPDATE venue_papers SET abstract=? WHERE dblp_key=?",
            (abstract, row["dblp_key"])
        )
        db.commit()
        status = "✅" if abstract else "❌"
        print(f"  [{i+1}/{len(rows)}] {status} {row['title'][:55]}...")
        time.sleep(0.35)

    db.close()
    print("[venue_monitor] 摘要补全完成")


# ──────────────────────────────────────────────────────────────
# AI 翻译（knot-cli，复用 paper_monitor_cache.db 的翻译缓存）
# ──────────────────────────────────────────────────────────────

import hashlib
import subprocess


def _ai_translate(text: str) -> str:
    """用 knot-cli 做 AI 翻译，失败返回原文。"""
    prompt = (
        "请将以下英文学术论文摘要翻译成中文，保留专业术语（如 KV cache、LLM、Transformer 等）。"
        "只输出翻译结果，不要任何解释：\n\n" + text[:600]
    )
    try:
        result = subprocess.run(
            ["knot-cli", "chat", "-p", prompt, "-o", "json"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            translated = data.get("response", "").strip()
            if translated and len(translated) > 5:
                return translated
    except Exception as e:
        print(f"[venue_monitor] knot-cli 翻译失败: {e}")
    return text


def translate_abstracts(years: Optional[list[int]] = None):
    """
    对命中论文中有摘要但未翻译的，批量做 AI 翻译（带缓存）。
    """
    if years is None:
        years = [datetime.now().year]

    db = _get_db()

    # 确保列存在
    cols = [r[1] for r in db.execute("PRAGMA table_info(venue_papers)").fetchall()]
    if "abstract_zh" not in cols:
        db.execute("ALTER TABLE venue_papers ADD COLUMN abstract_zh TEXT DEFAULT ''")
        db.commit()

    placeholders = ",".join("?" * len(years))
    rows = db.execute(f"""
        SELECT dblp_key, abstract FROM venue_papers
        WHERE year IN ({placeholders})
          AND matched_kws != '[]' AND matched_kws != ''
          AND abstract IS NOT NULL AND abstract != ''
          AND (abstract_zh IS NULL OR abstract_zh = '')
    """, years).fetchall()

    print(f"[venue_monitor] 待翻译: {len(rows)} 篇")
    for i, row in enumerate(rows):
        src_hash = hashlib.md5(row["abstract"].encode()).hexdigest()

        # 先查翻译缓存
        cached = db.execute(
            "SELECT translated FROM translation_cache WHERE src_hash=?",
            (src_hash,)
        ).fetchone()

        if cached:
            zh = cached["translated"]
            print(f"  [{i+1}/{len(rows)}] 💾 缓存命中")
        else:
            print(f"  [{i+1}/{len(rows)}] 🤖 AI翻译: {row['abstract'][:40]}...")
            zh = _ai_translate(row["abstract"])
            now = datetime.now().isoformat()
            try:
                db.execute(
                    """INSERT OR REPLACE INTO translation_cache
                       (src_hash, src_text, translated, engine, cached_at)
                       VALUES (?,?,?,?,?)""",
                    (src_hash, row["abstract"][:500], zh, "knot-ai", now)
                )
            except Exception:
                pass
            time.sleep(1)

        db.execute(
            "UPDATE venue_papers SET abstract_zh=? WHERE dblp_key=?",
            (zh, row["dblp_key"])
        )
        db.commit()

    db.close()
    print("[venue_monitor] 翻译完成")


# ──────────────────────────────────────────────────────────────
# 独立 Markdown 月报生成
# ──────────────────────────────────────────────────────────────

TOPIC_ORDER = [
    "🚀 LLM推理优化", "🏗️ AI Infra / 推理框架",
    "🔵 DeepSeek 相关", "🟣 Qwen 相关",
    "💾 KV Cache 优化", "⚡ 推测解码 / 量化",
    "🧠 Transformer / Attention", "📦 内存 / 并行优化",
    "🔀 调度 / 资源管理", "🌐 通用 AI Infra",
]


def generate_markdown_report(
    years: Optional[list[int]] = None,
    output_path: Optional[str] = None,
    limit_per_venue: int = 999,
) -> str:
    """生成顶会论文 Markdown 月报，含摘要翻译。"""
    if years is None:
        years = [datetime.now().year]

    db = _get_db()

    # 确保列存在
    cols = [r[1] for r in db.execute("PRAGMA table_info(venue_papers)").fetchall()]
    has_abstract = "abstract" in cols
    has_abstract_zh = "abstract_zh" in cols

    placeholders = ",".join("?" * len(years))
    rows = db.execute(f"""
        SELECT * FROM venue_papers
        WHERE year IN ({placeholders})
          AND matched_kws != '[]' AND matched_kws != ''
        ORDER BY venue, year DESC, title
    """, years).fetchall()
    db.close()

    if not rows:
        return "# 顶会论文监控月报\n\n> 暂无命中关键词的论文。"

    from collections import defaultdict
    topic_groups = defaultdict(list)
    venue_stats = defaultdict(int)

    for row in rows:
        kws = json.loads(row["matched_kws"]) if row["matched_kws"] else []
        # 根据命中关键词推断主题（简化内联映射）
        topic = "🌐 通用 AI Infra"
        kws_lower = [kw.lower() for kw in kws]
        if any(k in kws_lower for k in ["kv cache", "kv-cache", "kvcache", "attention cache", "prefix cache", "key-value cache"]):
            topic = "💾 KV Cache 优化"
        elif any(k in kws_lower for k in ["speculative decoding", "quantization", "model compression", "network quantization", "gptq"]):
            topic = "⚡ 推测解码 / 量化"
        elif any(k in kws_lower for k in ["flash attention", "paged attention", "linear attention", "transformer"]):
            topic = "🧠 Transformer / Attention"
        elif any(k in kws_lower for k in ["model parallelism", "tensor parallelism", "pipeline parallelism", "memory management", "memory efficiency", "hybrid memory"]):
            topic = "📦 内存 / 并行优化"
        elif any(k in kws_lower for k in ["gpu scheduling", "cluster scheduling", "resource management", "load balancing", "continuous batching", "runtime scheduling"]):
            topic = "🔀 调度 / 资源管理"
        elif any(k in kws_lower for k in ["vllm", "sglang", "tensorrt-llm", "tensorrt", "inference framework", "model serving", "disaggregated serving", "disaggregated inference"]):
            topic = "🏗️ AI Infra / 推理框架"
        elif any(k in kws_lower for k in ["deepseek"]):
            topic = "🔵 DeepSeek 相关"
        elif any(k in kws_lower for k in ["qwen"]):
            topic = "🟣 Qwen 相关"
        elif any(k in kws_lower for k in ["llm inference", "inference optimization", "serving system", "llm serving", "large language model inference"]):
            topic = "🚀 LLM推理优化"
        topic_groups[topic].append((row, kws))
        venue_stats[row["venue"]] += 1

    year_str = "/".join(str(y) for y in years)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# 🎓 顶会论文监控月报 ({year_str})",
        f"\n> 生成时间：{now_str}  |  数据来源：DBLP + Semantic Scholar\n",
        "## 📊 总览\n",
        "| 会议 | 命中论文数 |",
        "|------|-----------|",
    ]
    for vname, _ in VENUES:
        cnt = venue_stats.get(vname, 0)
        if cnt > 0:
            lines.append(f"| **{vname}** | {cnt} |")
    lines.append(f"| **合计** | **{len(rows)}** |")
    lines.append("")

    lines.append("## 📑 按主题分类\n")
    sorted_topics = sorted(
        topic_groups.keys(),
        key=lambda t: TOPIC_ORDER.index(t) if t in TOPIC_ORDER else 999
    )

    for topic in sorted_topics:
        papers = topic_groups[topic]
        lines.append(f"### {topic} ({len(papers)} 篇)\n")
        for row, kws in sorted(papers, key=lambda x: x[0]["venue"]):
            title   = row["title"]
            venue   = row["venue"]
            year    = row["year"]
            url     = row["doi_url"] or row["dblp_url"] or ""
            authors = row["authors"] or ""
            kw_str  = ", ".join(kws[:3])

            if url:
                lines.append(f"**[{title}]({url})**")
            else:
                lines.append(f"**{title}**")
            lines.append(f"*{venue} {year}* | 关键词：`{kw_str}`")
            if authors:
                lines.append(f"作者：{authors}")

            if has_abstract_zh and row["abstract_zh"]:
                lines.append(f"\n> {row['abstract_zh'][:300]}")
            elif has_abstract and row["abstract"]:
                lines.append(f"\n> {row['abstract'][:200]}...")
            lines.append("")

    lines.append("---\n")
    lines.append("## 📋 完整列表（按会议）\n")
    for vname, _ in VENUES:
        vpapers = [(r, kws) for r, kws in
                   [(row, json.loads(row["matched_kws"])) for row in rows if row["venue"] == vname]
                   if kws]
        if not vpapers:
            continue
        lines.append(f"### {vname} ({len(vpapers)} 篇)\n")
        lines.append("| 标题 | 关键词 | 链接 |")
        lines.append("|------|--------|------|")
        for row, kws in vpapers:
            t = row["title"].replace("|", "\\|")
            k = ", ".join(kws[:2]).replace("|", "\\|")
            url = row["doi_url"] or row["dblp_url"] or ""
            link = f"[🔗]({url})" if url else "-"
            lines.append(f"| {t[:65]} | `{k}` | {link} |")
        lines.append("")

    report = "\n".join(lines)

    if output_path is None:
        base = Path(__file__).parent
        output_path = str(base / f"venue_report_{year_str.replace('/', '_')}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[venue_monitor] 报告已保存：{output_path}")
    return report


# ──────────────────────────────────────────────────────────────
# CLI 入口（独立运行时刷新并打印统计）
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="顶会论文监控 - DBLP + S2 + AI翻译")
    parser.add_argument("--years", nargs="+", type=int,
                        default=[datetime.now().year],
                        help="年份，默认当前年")
    parser.add_argument("--force", action="store_true", help="强制重新拉取（忽略缓存）")
    parser.add_argument("--abstract", action="store_true", help="补全 Semantic Scholar 摘要")
    parser.add_argument("--translate", action="store_true", help="AI 翻译摘要")
    parser.add_argument("--report", action="store_true", help="生成 Markdown 月报")
    parser.add_argument("--report-only", action="store_true", help="只生成报告，不重新抓取")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    parser.add_argument("--venues", nargs="+", default=None, help="指定会议，默认全部")
    args = parser.parse_args()

    target_venues = VENUES
    if args.venues:
        names_upper = [v.upper() for v in args.venues]
        target_venues = [(n, s) for n, s in VENUES if n.upper() in names_upper]

    years = args.years

    if not args.report_only:
        print(f"[venue_monitor] 开始刷新顶会论文缓存（years={years}, force={args.force}）")
        db = _get_db()
        stats: dict[str, int] = {}
        for venue_name, stream in target_venues:
            total_matched = 0
            for year in years:
                cnt = fetch_and_cache_venue(venue_name, stream, year, db, force=args.force)
                total_matched += cnt
                time.sleep(0.3)
            stats[venue_name] = total_matched
            print(f"[venue_monitor] {venue_name}: {total_matched} matched papers cached")
        db.close()

        print("\n=== 匹配统计 ===")
        for v, cnt in stats.items():
            print(f"  {v:10s}: {cnt} 篇命中")
        print(f"\n总计命中: {sum(stats.values())} 篇")

    if args.abstract:
        enrich_abstracts(years=years)

    if args.translate:
        translate_abstracts(years=years)

    if args.report or args.report_only:
        report = generate_markdown_report(years=years, output_path=args.output)
        lines = report.split("\n")
        print("\n".join(lines[:80]))
        if len(lines) > 80:
            print(f"\n... 共 {len(lines)} 行，完整内容已保存到文件。")
    else:
        # 默认打印命中详情
        matched = get_matched_papers(years=years, limit_per_venue=10)
        print("\n=== 命中论文详情 ===")
        for venue_name, papers in matched.items():
            if not papers:
                continue
            print(f"\n【{venue_name}】")
            for p in papers:
                kws = ", ".join(p["matched_kws"][:3])
                print(f"  [{p['year']}] {p['title'][:70]}")
                print(f"         关键词: {kws}")
