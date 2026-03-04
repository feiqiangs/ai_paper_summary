#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Infra 论文监控脚本
监控方向：LLM推理优化、AI Infra、DeepSeek、KV Cache
"""

import sys
import os
import re
import time
import requests
from datetime import datetime

# 将 arxiv_search 所在目录加入 path
ARXIV_SKILL_DIR = "/data/workspace/.agent/skills/arxiv-paper-search/scripts"
sys.path.insert(0, ARXIV_SKILL_DIR)

from arxiv_search import ArxivSearcher

# 监控主题配置
TOPICS = [
    {
        "name": "🚀 LLM推理优化 (Inference Optimization)",
        "keywords": ["LLM inference", "inference optimization", "serving system", "agentic inference", "disaggregated inference", "I/O bottleneck LLM"],
        "days": 7,
        "num": 8,
    },
    {
        "name": "🏗️ AI Infra / 推理框架",
        "keywords": ["vLLM", "SGLang", "TensorRT-LLM", "inference framework", "storage bandwidth LLM", "disaggregated LLM"],
        "days": 7,
        "num": 8,
    },
    {
        "name": "🔵 DeepSeek 相关",
        "keywords": ["DeepSeek"],
        "days": 7,
        "num": 8,
    },
    {
        "name": "🟣 Qwen 相关",
        "keywords": ["Qwen"],
        "days": 7,
        "num": 6,
    },
    {
        "name": "💾 KV Cache 优化",
        "keywords": ["KV cache", "attention cache", "prefix cache"],
        "days": 7,
        "num": 8,
    },
    {
        "name": "⚡ 推测解码 / 量化",
        "keywords": ["speculative decoding", "quantization LLM", "model compression inference"],
        "days": 7,
        "num": 6,
    },
]

# 已知机构关键词映射（用于从摘要/作者信息中识别机构）
KNOWN_ORGS = [
    # 大厂
    "Google", "DeepMind", "Google DeepMind",
    "Meta", "Meta AI", "FAIR",
    "Microsoft", "Microsoft Research",
    "OpenAI",
    "Anthropic",
    "Amazon", "AWS",
    "Apple",
    "NVIDIA",
    "Intel",
    "IBM", "IBM Research",
    "Baidu", "ByteDance", "Alibaba", "Tencent", "Huawei",
    "DeepSeek",
    # 高校
    "MIT", "Stanford", "Carnegie Mellon", "CMU", "Berkeley", "UC Berkeley",
    "Harvard", "Princeton", "Yale", "Columbia", "Cornell", "Caltech",
    "University of Washington", "UW", "UIUC", "UT Austin",
    "Oxford", "Cambridge", "ETH Zurich", "EPFL",
    "Tsinghua", "Peking University", "PKU", "Fudan", "Zhejiang University",
    "SJTU", "Shanghai Jiao Tong", "HKUST", "NUS", "NTU",
    # 研究机构
    "Allen Institute", "AI2",
    "Hugging Face",
    "EleutherAI",
    "Together AI",
    "Mistral",
    "Cohere",
]

def extract_affiliation(summary: str, authors: list) -> str:
    """
    从摘要文本中提取机构信息。
    arXiv API 不提供 affiliation 字段，尝试从摘要中识别已知机构名称。
    若无法识别，则返回第一作者姓名。
    """
    found = []
    text = summary
    for org in KNOWN_ORGS:
        # 使用词边界匹配，避免误匹配子字符串
        pattern = r'\b' + re.escape(org) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found.append(org)

    # 去重并保留顺序
    seen = set()
    unique = []
    for org in found:
        key = org.lower()
        if key not in seen:
            seen.add(key)
            unique.append(org)

    if unique:
        return ", ".join(unique[:3])  # 最多显示3个机构

    # 识别不到机构时，返回第一作者姓名
    if authors:
        return authors[0]
    return "未知"


# 翻译缓存，避免重复请求
_translate_cache: dict = {}

def translate_to_chinese(text: str) -> str:
    """
    将英文文本翻译成中文，使用 MyMemory 免费翻译接口。
    带缓存和异常兜底（失败时返回原文截断版本）。
    """
    if not text:
        return text

    # 命中缓存直接返回
    if text in _translate_cache:
        return _translate_cache[text]

    try:
        for attempt in range(2):
            resp = requests.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text, "langpair": "en|zh"},
                timeout=10,
            )
            result = resp.json()
            translated = result.get("responseData", {}).get("translatedText", "")
            if translated and translated != text:
                _translate_cache[text] = translated
                # 避免请求过快被限速
                time.sleep(0.5)
                return translated
            # 第一次失败，等待后重试
            time.sleep(1.5)
    except Exception:
        pass

    # 翻译失败时返回原文
    _translate_cache[text] = text
    return text


def summarize_and_translate(summary: str) -> str:
    """
    用一句话总结论文摘要，然后翻译成中文。
    取前2句话作为核心内容进行翻译，得到简短中文总结。
    """
    summary = summary.strip().replace('\n', ' ')
    # 按 '. ' 分句，取前2句作为核心摘要
    parts = summary.split('. ')
    if len(parts) >= 2:
        core = '. '.join(parts[:2]) + '.'
    else:
        core = summary[:300]
    # 限制长度避免翻译接口超限
    return translate_to_chinese(core[:350])


def format_summary_dual(summary: str) -> tuple:
    """
    返回 (short_zh, original_en) 两个值：
    - short_zh: 前两句总结后翻译成中文（用于表格展示）
    - original_en: 原文英文 Abstract（用于 hover tooltip 显示）
    """
    summary = summary.strip().replace('\n', ' ')
    short_zh = summarize_and_translate(summary)
    # hover 显示原始英文 Abstract，不翻译
    original_en = summary
    return short_zh, original_en


def format_summary(summary: str, max_len: int = 400) -> str:
    """截取摘要核心句子并翻译为中文（兼容旧调用）"""
    summary = summary.strip().replace('\n', ' ')
    short = summary[:max_len]
    return translate_to_chinese(short)


# 所有热度关键词（命中越多分越高）
HOT_KEYWORDS = [
    "inference", "serving", "throughput", "latency", "memory", "bandwidth",
    "kv cache", "speculative", "quantization", "disaggregated", "prefill",
    "decode", "vllm", "sglang", "tensorrt", "deepseek", "moe", "attention",
    "transformer", "llm", "gpu", "pipeline", "parallelism", "token",
]

# 关键字候选词库（用于从标题/摘要中提取论文关键字）
KEYWORD_CANDIDATES = [
    "LLM", "inference", "serving", "KV cache", "speculative decoding",
    "quantization", "attention", "transformer", "GPU", "memory",
    "throughput", "latency", "bandwidth", "prefill", "decode",
    "disaggregated", "pipeline", "parallelism", "MoE", "token",
    "vLLM", "SGLang", "TensorRT", "DeepSeek", "RAG", "agent",
    "long context", "compression", "pruning", "distillation",
    "multi-head", "low-rank", "flash attention", "paged attention",
    "continuous batching", "prefix caching", "request scheduling",
    "model parallelism", "tensor parallelism", "CUDA", "SRAM", "DRAM",
    "privacy", "confidential", "federated", "reinforcement learning",
    "chain-of-thought", "reasoning", "embodied", "multimodal", "VLM",
    "ASR", "OCR", "code generation", "benchmark",
]

def extract_keywords(title: str, summary: str, top_n: int = 5) -> str:
    """从标题和摘要中提取最相关的关键字，返回逗号分隔的字符串"""
    text = (title + " " + summary).lower()
    hits = []
    for kw in KEYWORD_CANDIDATES:
        if kw.lower() in text:
            hits.append(kw)
    # 优先取标题中命中的
    title_lower = title.lower()
    title_hits = [kw for kw in hits if kw.lower() in title_lower]
    other_hits = [kw for kw in hits if kw not in title_hits]
    ordered = title_hits + other_hits
    return ", ".join(ordered[:top_n]) if ordered else "LLM"

# ---- HTML experimental 提取（优先方案） ----

# HTML 提取结果缓存：{clean_id: (abstract, keywords)}
_html_cache: dict = {}


def fetch_arxiv_html(paper_id: str) -> str:
    """
    获取 arXiv HTML (experimental) 页面内容。
    URL 格式：https://arxiv.org/html/{clean_id}
    返回 HTML 字符串，失败返回空字符串。
    """
    clean_id = re.sub(r'v\d+$', '', paper_id)
    html_url = f"https://arxiv.org/html/{clean_id}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PaperMonitor/1.0; +https://github.com/feiqiangs/ai_paper_summary)",
            "Accept": "text/html,application/xhtml+xml",
        }
        resp = requests.get(html_url, headers=headers, timeout=20)
        if resp.status_code == 200 and 'html' in resp.headers.get('Content-Type', '').lower():
            return resp.text
    except Exception as e:
        print(f"⚠️ HTML 获取失败 [{paper_id}]: {e}", file=sys.stderr)
    return ""


def extract_from_html(html_text: str) -> tuple:
    """
    从 arXiv HTML (experimental) 页面提取 Abstract 和 Keywords。
    返回 (abstract_text, keywords_list)。
    """
    from html.parser import HTMLParser

    abstract = ""
    keywords = []

    # ---- 用正则从 HTML 中提取结构化内容 ----

    # 1. 提取 Abstract
    # arXiv HTML 中 abstract 通常在 <section id="abstract"> 或 <div class="ltx_abstract"> 中
    abstract_patterns = [
        # <section id="abstract">...<p>...</p>...</section>
        r'<section[^>]*id=["\']abstract["\'][^>]*>([\s\S]{50,5000}?)</section>',
        # <div class="ltx_abstract">...<p class="ltx_p">...</p>...</div>
        r'<div[^>]*class=["\'][^"\']*ltx_abstract[^"\']*["\'][^>]*>([\s\S]{50,5000}?)</div>',
        # <blockquote class="abstract"> (旧版 arXiv)
        r'<blockquote[^>]*class=["\'][^"\']*abstract[^"\']*["\'][^>]*>([\s\S]{50,3000}?)</blockquote>',
    ]
    for pat in abstract_patterns:
        m = re.search(pat, html_text, re.IGNORECASE)
        if m:
            raw = m.group(1)
            # 去掉 HTML 标签
            raw = re.sub(r'<[^>]+>', ' ', raw)
            # 去掉多余空白
            raw = re.sub(r'\s+', ' ', raw).strip()
            # 去掉 "Abstract" 标题字样
            raw = re.sub(r'^[Aa]bstract[\s:.\-–—]*', '', raw).strip()
            if len(raw) > 80:
                abstract = raw
                break

    # 2. 提取 Keywords
    # arXiv HTML 中 keywords 通常在 <div class="ltx_keywords"> 或含 "Keywords" 文字的段落中
    kw_patterns = [
        r'<div[^>]*class=["\'][^"\']*ltx_keywords[^"\']*["\'][^>]*>([\s\S]{5,500}?)</div>',
        r'<p[^>]*class=["\'][^"\']*ltx_keywords[^"\']*["\'][^>]*>([\s\S]{5,500}?)</p>',
        # 含 "Keywords:" 文字的段落
        r'(?i)[Kk]ey\s*[Ww]ords?\s*[:\-–—]\s*([\s\S]{5,400}?)(?=</p>|</div>|<br|<h)',
        r'(?i)[Ii]ndex\s+[Tt]erms?\s*[:\-–—]\s*([\s\S]{5,400}?)(?=</p>|</div>|<br|<h)',
    ]
    for pat in kw_patterns:
        m = re.search(pat, html_text, re.IGNORECASE)
        if m:
            raw = m.group(1)
            # 去掉 HTML 标签
            raw = re.sub(r'<[^>]+>', ' ', raw)
            raw = re.sub(r'\s+', ' ', raw).strip()
            # 去掉 "Keywords:" 前缀
            raw = re.sub(r'^[Kk]ey\s*[Ww]ords?\s*[:\-–—]\s*', '', raw).strip()
            raw = re.sub(r'^[Ii]ndex\s+[Tt]erms?\s*[:\-–—]\s*', '', raw).strip()
            # 按分隔符拆分
            kws = re.split(r'[,;·•]\s*', raw)
            kws = [k.strip().strip('.,;·•-–—') for k in kws]
            kws = [k for k in kws if k and 2 <= len(k) <= 60 and not k.isdigit()]
            kws = [k for k in kws if k.count(' ') <= 5]
            if kws:
                keywords = kws[:8]
                break

    return abstract, keywords


def get_paper_content_from_html(paper: dict) -> tuple:
    """
    优先通过 arXiv HTML (experimental) 链接提取 Abstract 和 Keywords。
    失败时 fallback 到 arXiv API 返回的 summary。
    返回 (abstract_text, keywords_list)。
    """
    paper_id = paper["id"]
    clean_id = re.sub(r'v\d+$', '', paper_id)

    # 命中缓存
    if clean_id in _html_cache:
        return _html_cache[clean_id]

    print(f"  🌐 获取 HTML [{clean_id}] ...", file=sys.stderr)

    html_text = fetch_arxiv_html(paper_id)
    if html_text:
        abstract, keywords = extract_from_html(html_text)
        # 如果 HTML 提取的 abstract 太短，fallback 到 API 摘要
        if len(abstract) < 80:
            print(f"  ⚠️ HTML abstract 太短，fallback 到 API summary [{clean_id}]", file=sys.stderr)
            abstract = paper["summary"]
        result = (abstract, keywords)
    else:
        print(f"  ⚠️ HTML 不可用，使用 API summary [{clean_id}]", file=sys.stderr)
        result = (paper["summary"], [])

    _html_cache[clean_id] = result
    # 避免请求过快
    time.sleep(0.3)
    return result


# 保留旧函数名作为别名，供 build_echarts_wordcloud_data 等调用
_pdf_cache = _html_cache


def get_paper_content_from_pdf(paper: dict) -> tuple:
    """兼容旧调用，实际转发到 HTML 提取方案"""
    return get_paper_content_from_html(paper)


# ---- 以下 PDF 相关函数保留但不再主动调用 ----

def extract_abstract_and_keywords_from_pdf(pdf_path: str) -> tuple:
    """
    从 PDF 文件中提取 Abstract 和 Keywords 章节内容。
    优先使用 pdftotext（poppler-utils）提取前5页文本，fallback 到 pdfplumber。
    返回 (abstract_text, keywords_list) 元组。
    """
    import subprocess

    full_text = ""

    # 优先用 pdftotext（文本质量更好，单词间距正确）
    try:
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "5", pdf_path, "-"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and len(result.stdout.strip()) > 100:
            full_text = result.stdout
    except Exception:
        pass

    # fallback：pdfplumber
    if not full_text.strip():
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:5]:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
        except Exception as e:
            print(f"⚠️ PDF 解析失败 [{pdf_path}]: {e}", file=sys.stderr)

    if not full_text.strip():
        return "", []

    abstract = ""
    keywords = []

    # ---- 提取 Abstract ----
    # 注意：部分论文 PDF 中 ABSTRACT 渲染为 "A BSTRACT" 或 "A B S T R A C T" 等形式
    abstract_patterns = [
        # 处理字母间带空格的大写标题，如 "A BSTRACT" / "A B S T R A C T"
        r'(?i)(?:^|\n)\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*\n([\s\S]{80,3000}?)(?=\n\s*(?:\d+\.?\s+introduction|\d+\s+introduction|keywords?|index terms?|\n\s*\n\s*\n))',
        # 普通 Abstract 标题
        r'(?i)(?:^|\n)\s*abstract\s*\n([\s\S]{80,3000}?)(?=\n\s*(?:\d+\.?\s+introduction|keywords?|index terms?))',
        # Abstract: 后面跟内容
        r'(?i)\babstract\b[:\s—–-]+\n?([\s\S]{80,2000}?)(?=\n\s*(?:\d+\.?\s+introduction|keywords?|index terms?))',
        # 简单：abstract 后的大段文字（宽松匹配）
        r'(?i)a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*\n([\s\S]{80,2000}?)(?=\n\s*\n\s*\n)',
    ]
    for pat in abstract_patterns:
        m = re.search(pat, full_text)
        if m:
            raw = m.group(1).strip()
            raw = re.sub(r'\n+', ' ', raw).strip()
            raw = re.sub(r'\s{2,}', ' ', raw)
            if len(raw) > 80:
                abstract = raw
                break

    # 双栏论文特殊处理：Abstract 内容可能在标题之前（pdftotext 按列顺序提取）
    if not abstract:
        # 找到 "Abstract" 标题的位置
        m_pos = re.search(r'(?i)\n\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*\n', full_text)
        if m_pos:
            before = full_text[:m_pos.start()]
            # 找所有段落（按双换行分隔），取字符数最多的段落作为摘要候选
            parts = re.split(r'\n\s*\n', before)
            # 过滤掉太短的段落（图表数字等）
            candidates = []
            for p in parts:
                p_clean = re.sub(r'\n+', ' ', p).strip()
                p_clean = re.sub(r'\s{2,}', ' ', p_clean)
                # 有效段落：长度 > 80，且单词数量合理（不是纯数字/符号）
                words = p_clean.split()
                real_words = [w for w in words if re.search(r'[a-zA-Z]{3,}', w)]
                if len(p_clean) > 80 and len(real_words) >= 10:
                    candidates.append(p_clean)
            if candidates:
                # 取最长的候选段落
                abstract = max(candidates, key=len)

    # ---- 提取 Keywords ----
    kw_patterns = [
        r'(?i)[Kk]ey\s*[Ww]ords?\s*[:：—–-]\s*([^\n]{5,300})',
        r'(?i)[Ii]ndex\s+[Tt]erms?\s*[:：—–-]\s*([^\n]{5,300})',
        r'(?i)[Kk]ey\s*[Ww]ords?\s*\n([^\n]{5,300})',
        r'(?i)[Kk]ey\s*[Ww]ords?\s*[:：]\s*([\s\S]{5,300}?)(?=\n\s*\n|\n\s*[A-Z])',
    ]
    for pat in kw_patterns:
        m = re.search(pat, full_text)
        if m:
            raw = m.group(1).strip()
            kws = re.split(r'[,;，；·•]\s*', raw)
            kws = [k.strip().strip('.-·•') for k in kws]
            kws = [k for k in kws if k and 2 <= len(k) <= 60 and not k.isdigit()]
            kws = [k for k in kws if k.count(' ') <= 5]
            if kws:
                keywords = kws[:8]
                break

    return abstract, keywords


def extract_paper_keywords(summary: str) -> list:
    """
    从论文摘要文本中提取 Keywords 字段内容（兼容旧调用）。
    仅作为 PDF 提取失败时的 fallback。
    """
    patterns = [
        r'[Kk]eywords?\s*[:：]\s*([^\n.]+)',
        r'[Ii]ndex [Tt]erms?\s*[:：]\s*([^\n.]+)',
        r'[Kk]ey [Ww]ords?\s*[:：]\s*([^\n.]+)',
    ]
    for pat in patterns:
        m = re.search(pat, summary)
        if m:
            raw = m.group(1).strip()
            kws = [k.strip().strip('•·') for k in re.split(r'[,;，；]', raw)]
            kws = [k for k in kws if k and len(k) > 1 and len(k) < 50]
            if kws:
                return kws[:8]
    return []


def extract_keyphrases_en(text: str, title: str = "", top_n: int = 5) -> list:
    """
    当论文没有 Keywords 章节时，从 Abstract（+标题）中提取 top_n 个英文关键短语。
    全程英文，不翻译，短语能直观反映论文核心技术贡献。

    策略：
    1. 优先匹配领域术语短语词典（2-4词，覆盖 AI/ML 核心概念）
    2. 标题词加权（标题词更代表论文核心）
    3. 提取高频技术名词作为补充
    4. 去重、去子串，返回格式化的英文短语
    """
    # 停用词（过滤掉无实义的功能词）
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'can', 'not', 'no', 'nor',
        'so', 'yet', 'both', 'either', 'neither', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'than', 'then', 'that', 'this', 'these', 'those',
        'which', 'who', 'whom', 'what', 'when', 'where', 'why', 'how', 'all',
        'any', 'if', 'as', 'its', 'it', 'we', 'our', 'they', 'their', 'also',
        'while', 'into', 'about', 'up', 'out', 'over', 'after', 'between',
        'through', 'during', 'before', 'under', 'among', 'i', 'he', 'she',
        # 论文功能词（动词/形容词/副词，无关键词价值）
        'paper', 'work', 'approach', 'method', 'propose', 'proposed', 'present',
        'show', 'shows', 'demonstrate', 'demonstrates', 'achieve', 'achieves',
        'improve', 'improves', 'existing', 'new', 'novel', 'based', 'using',
        'used', 'use', 'thus', 'however', 'therefore', 'moreover', 'furthermore',
        'result', 'results', 'experiment', 'experiments', 'evaluation',
        'significant', 'significantly', 'compared', 'across', 'multiple',
        'various', 'different', 'two', 'three', 'one', 'first', 'second',
        'third', 'high', 'low', 'large', 'small', 'further', 'recent',
        'current', 'without', 'within', 'state', 'art', 'well', 'find',
        'found', 'make', 'made', 'take', 'taken', 'give', 'given', 'need',
        'needs', 'able', 'even', 'only', 'just', 'much', 'many', 'several',
        'show', 'shown', 'demonstrate', 'demonstrated', 'enable', 'enables',
        'task', 'tasks', 'system', 'systems', 'data', 'dataset', 'datasets',
        'set', 'sets', 'number', 'numbers', 'type', 'types', 'level', 'levels',
        'way', 'ways', 'time', 'times', 'process', 'processing', 'general',
        'specific', 'important', 'effective', 'efficient', 'better', 'best',
        'good', 'problem', 'challenge', 'solution', 'address', 'addresses',
        'introduce', 'introduces', 'introduced', 'introduction',
        'state-of-the-art', 'effectively', 'benchmark', 'benchmarks',
        'performance', 'outperform', 'outperforms', 'extensive', 'comprehensive',
        'robust', 'scalable', 'flexible', 'demonstrate', 'indicate', 'suggest',
        'apply', 'applied', 'utilize', 'utilized', 'leverage', 'leverages',
        'train', 'trained', 'training', 'test', 'testing', 'evaluate',
        # 标题中常见的无实义词
        'technical', 'report', 'survey', 'study', 'analysis', 'via', 'towards',
        'efficient', 'fast', 'better', 'improved', 'enhanced', 'advanced',
        'simple', 'unified', 'end', 'end-to-end', 'beyond', 'revisiting',
    }

    # AI/ML 领域高价值术语短语词典（按类别分组，覆盖核心技术方向）
    # 格式：(phrase_lower, display_form)
    DOMAIN_PHRASES = [
        # ── LLM 核心架构 ──
        ('large language model', 'Large Language Model'),
        ('vision language model', 'Vision-Language Model'),
        ('multimodal large language model', 'Multimodal LLM'),
        ('multimodal model', 'Multimodal Model'),
        ('foundation model', 'Foundation Model'),
        ('autoregressive model', 'Autoregressive Model'),
        ('diffusion model', 'Diffusion Model'),
        ('mixture of experts', 'Mixture of Experts'),
        ('state space model', 'State Space Model'),
        ('graph neural network', 'Graph Neural Network'),
        ('neural network', 'Neural Network'),
        ('transformer', 'Transformer'),
        # ── 推理/服务优化 ──
        ('kv cache', 'KV Cache'),
        ('kv cache compression', 'KV Cache Compression'),
        ('speculative decoding', 'Speculative Decoding'),
        ('flash attention', 'Flash Attention'),
        ('grouped query attention', 'Grouped Query Attention'),
        ('multi-head attention', 'Multi-Head Attention'),
        ('paged attention', 'Paged Attention'),
        ('continuous batching', 'Continuous Batching'),
        ('prefix caching', 'Prefix Caching'),
        ('inference efficiency', 'Inference Efficiency'),
        ('inference acceleration', 'Inference Acceleration'),
        ('token generation', 'Token Generation'),
        ('long context', 'Long Context'),
        ('context window', 'Context Window'),
        ('context length', 'Context Length'),
        # ── 训练/微调 ──
        ('reinforcement learning from human feedback', 'RLHF'),
        ('reinforcement learning', 'Reinforcement Learning'),
        ('supervised fine-tuning', 'Supervised Fine-Tuning'),
        ('instruction tuning', 'Instruction Tuning'),
        ('parameter-efficient fine-tuning', 'Parameter-Efficient Fine-Tuning'),
        ('parameter efficient fine-tuning', 'Parameter-Efficient Fine-Tuning'),
        ('knowledge distillation', 'Knowledge Distillation'),
        ('continual learning', 'Continual Learning'),
        ('self-supervised learning', 'Self-Supervised Learning'),
        ('contrastive learning', 'Contrastive Learning'),
        ('direct preference optimization', 'Direct Preference Optimization'),
        ('proximal policy optimization', 'PPO'),
        ('lora', 'LoRA'),
        ('rlhf', 'RLHF'),
        ('dpo', 'DPO'),
        # ── 推理能力 ──
        ('chain-of-thought', 'Chain-of-Thought'),
        ('chain of thought', 'Chain-of-Thought'),
        ('reasoning ability', 'Reasoning Ability'),
        ('mathematical reasoning', 'Mathematical Reasoning'),
        ('logical reasoning', 'Logical Reasoning'),
        ('in-context learning', 'In-Context Learning'),
        ('in context learning', 'In-Context Learning'),
        ('few-shot learning', 'Few-Shot Learning'),
        ('few shot learning', 'Few-Shot Learning'),
        ('zero-shot', 'Zero-Shot'),
        # ── 应用场景 ──
        ('retrieval-augmented generation', 'RAG'),
        ('retrieval augmented generation', 'RAG'),
        ('code generation', 'Code Generation'),
        ('text generation', 'Text Generation'),
        ('image generation', 'Image Generation'),
        ('video generation', 'Video Generation'),
        ('speech recognition', 'Speech Recognition'),
        ('machine translation', 'Machine Translation'),
        ('question answering', 'Question Answering'),
        ('information extraction', 'Information Extraction'),
        ('named entity recognition', 'Named Entity Recognition'),
        ('autonomous agent', 'Autonomous Agent'),
        ('tool use', 'Tool Use'),
        ('function calling', 'Function Calling'),
        # ── 系统/效率 ──
        ('model compression', 'Model Compression'),
        ('quantization', 'Quantization'),
        ('weight pruning', 'Weight Pruning'),
        ('knowledge graph', 'Knowledge Graph'),
        ('distributed training', 'Distributed Training'),
        ('pipeline parallelism', 'Pipeline Parallelism'),
        ('tensor parallelism', 'Tensor Parallelism'),
        ('data parallelism', 'Data Parallelism'),
        ('memory efficiency', 'Memory Efficiency'),
        ('throughput optimization', 'Throughput Optimization'),
        ('gpu memory', 'GPU Memory'),
        # ── 安全/对齐 ──
        ('safety alignment', 'Safety Alignment'),
        ('red teaming', 'Red Teaming'),
        ('hallucination', 'Hallucination'),
        ('factuality', 'Factuality'),
        ('federated learning', 'Federated Learning'),
        ('differential privacy', 'Differential Privacy'),
    ]

    # 清理文本
    text_clean = text.strip().replace('\n', ' ')
    text_clean = re.sub(r'\s+', ' ', text_clean)
    text_lower = text_clean.lower()
    title_lower = title.lower()

    # 提取标题中有实义的词（过滤停用词），用于后续加权
    title_meaningful_words = {
        w.lower() for w in re.findall(r'\b[A-Za-z][A-Za-z\-]{2,}\b', title)
        if w.lower() not in STOP_WORDS and len(w) >= 3
    }

    found = []   # [(display_form, score)]
    covered_words = set()

    # 第一步：精确匹配领域短语词典（按长度降序，优先匹配长短语）
    for phrase_lower, display in sorted(DOMAIN_PHRASES, key=lambda x: len(x[0]), reverse=True):
        if phrase_lower in text_lower:
            # 计算权重：在 abstract 中出现次数 + 标题中出现 +5
            score = text_lower.count(phrase_lower)
            if phrase_lower in title_lower:
                score += 5
            found.append((display, score))
            for w in phrase_lower.split():
                covered_words.add(w)

    # 第二步：提取高频技术名词（未被领域短语覆盖的单词）
    TECH_SUFFIXES = (
        'tion', 'ment', 'ness', 'ity', 'ism', 'ence', 'ance',
        'ology', 'ture', 'ware', 'work', 'gram',
    )
    # 只过滤真正无实义的通用词
    EXTRA_STOP = {
        'generation', 'representation', 'computation', 'implementation',
        'combination', 'configuration', 'consideration', 'communication',
        'requirement', 'measurement', 'deployment',
        'difference', 'reference',
        'importance', 'relevance', 'accordance', 'appearance',
        'efficiency', 'accuracy', 'quality', 'ability', 'capability',
        'complexity', 'diversity', 'flexibility', 'scalability',
        'availability', 'reliability', 'stability', 'visibility',
    }
    words = re.findall(r'\b[A-Za-z][A-Za-z]{2,}\b', text_clean)
    word_freq: dict = {}
    for w in words:
        wl = w.lower()
        if wl in STOP_WORDS or wl in covered_words or wl in EXTRA_STOP:
            continue
        is_technical = wl.endswith(TECH_SUFFIXES)
        if is_technical:
            canonical = w[0].upper() + w[1:]
            word_freq[canonical] = word_freq.get(canonical, 0) + 1

    # 第三步：识别标题中的系统名/模型名（如 DeepSeek-V3, GPT-4, LLaMA2）
    # 特征：含数字或版本号，或全大写缩写（3-8字符）
    system_name_pattern = re.compile(
        r'\b([A-Z][A-Za-z]*(?:[-_]?(?:[Vv]\d+|\d+[A-Za-z]?|[A-Z]{2,}))+)\b'  # DeepSeek-V3, GPT-4
        r'|\b([A-Z]{2,8})\b'  # MLA, MoE, RAG, LLM
    )
    for m in system_name_pattern.finditer(title):
        name = (m.group(1) or m.group(2) or '').strip()
        nl = name.lower()
        if nl in STOP_WORDS or nl in covered_words or len(name) < 2:
            continue
        if not any(nl in p.lower() for p, _ in found):
            # 系统名/缩写加高权重（出现在标题中）
            found.append((name, 6))

    # 标题中出现的有实义词额外加权
    for w, freq in list(word_freq.items()):
        if w.lower() in title_meaningful_words:
            freq += 4
        if not any(w.lower() in p.lower() for p, _ in found):
            found.append((w, freq))

    # 按权重排序
    found.sort(key=lambda x: x[1], reverse=True)
    candidates = [p for p, _ in found]

    # 去重：避免子串重复（有 "Large Language Model" 就不要 "Language Model"）
    filtered = []
    candidates_lower = [c.lower() for c in candidates]
    for i, phrase in enumerate(candidates):
        phrase_l = phrase.lower()
        is_substring = any(
            phrase_l in other_l and phrase_l != other_l
            for other_l in candidates_lower
        )
        if not is_substring:
            filtered.append(phrase)
        if len(filtered) >= top_n:
            break

    return filtered[:top_n] if filtered else candidates[:top_n]


# 兼容旧调用名
def extract_keyphrases_zh(text: str, top_n: int = 5) -> list:
    """兼容旧调用，转发到英文关键短语提取"""
    return extract_keyphrases_en(text, top_n=top_n)


# 保留旧函数名兼容调用
def extract_nouns_top5(text: str) -> list:
    """兼容旧调用，转发到 extract_keyphrases_zh"""
    return extract_keyphrases_zh(text, top_n=5)


# ---- Semantic Scholar 引用数据缓存 ----
# {clean_arxiv_id: {"citationCount": int, "influentialCitationCount": int}}
_ss_metrics_cache: dict = {}

# ---- Hugging Face Papers 点赞数缓存 ----
# {clean_arxiv_id: {"upvotes": int, "comments": int}}
_hf_metrics_cache: dict = {}

# ---- arXiv 版本号缓存 ----
# {clean_arxiv_id: int}
_arxiv_version_cache: dict = {}


def fetch_hf_paper_metrics(arxiv_id: str) -> dict:
    """
    从 Hugging Face Papers API 获取论文点赞数、评论数、GitHub repo 和 star 数。
    arxiv_id: 干净的 arXiv ID（如 '2501.12599'）
    返回 {"upvotes": int, "comments": int, "githubRepo": str, "githubStars": int}
    """
    clean_id = re.sub(r'v\d+$', '', arxiv_id)
    if clean_id in _hf_metrics_cache:
        return _hf_metrics_cache[clean_id]
    try:
        url = f"https://huggingface.co/api/papers/{clean_id}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            result = {
                "upvotes": data.get("upvotes", 0) or 0,
                "comments": len(data.get("comments", [])) if isinstance(data.get("comments"), list) else 0,
                "githubRepo": data.get("githubRepo") or "",
                "githubStars": data.get("githubStars") or 0,
            }
            _hf_metrics_cache[clean_id] = result
            return result
        else:
            _hf_metrics_cache[clean_id] = {"upvotes": 0, "comments": 0, "githubRepo": "", "githubStars": 0}
            return {"upvotes": 0, "comments": 0, "githubRepo": "", "githubStars": 0}
    except Exception:
        _hf_metrics_cache[clean_id] = {"upvotes": 0, "comments": 0, "githubRepo": "", "githubStars": 0}
        return {"upvotes": 0, "comments": 0, "githubRepo": "", "githubStars": 0}


def get_arxiv_version(paper_id: str) -> int:
    """
    从 paper_id 中提取 arXiv 版本号（如 '2501.12599v3' -> 3）。
    无版本号时返回 1。
    """
    clean_id = re.sub(r'v\d+$', '', paper_id)
    if clean_id in _arxiv_version_cache:
        return _arxiv_version_cache[clean_id]
    m = re.search(r'v(\d+)$', paper_id)
    version = int(m.group(1)) if m else 1
    _arxiv_version_cache[clean_id] = version
    return version


def fetch_semantic_scholar_metrics(paper_ids: list) -> dict:
    """
    批量从 Semantic Scholar API 获取论文引用数据。
    paper_ids: arXiv ID 列表（如 ['2501.12599', '2412.19437']）
    返回 {clean_id: {"citationCount": int, "influentialCitationCount": int}}
    """
    if not paper_ids:
        return {}

    # 过滤掉已缓存的
    uncached = [re.sub(r'v\d+$', '', pid) for pid in paper_ids
                if re.sub(r'v\d+$', '', pid) not in _ss_metrics_cache]
    if not uncached:
        return {pid: _ss_metrics_cache[pid] for pid in
                [re.sub(r'v\d+$', '', p) for p in paper_ids] if pid in _ss_metrics_cache}

    result = {}
    # 分批请求，每批最多 100 个（API 限制）
    batch_size = 100
    for i in range(0, len(uncached), batch_size):
        batch = uncached[i:i + batch_size]
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/batch"
            params = {"fields": "citationCount,influentialCitationCount"}
            body = {"ids": [f"arXiv:{pid}" for pid in batch]}
            resp = requests.post(url, params=params, json=body, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                for pid, item in zip(batch, data):
                    metrics = {
                        "citationCount": item.get("citationCount", 0) if item else 0,
                        "influentialCitationCount": item.get("influentialCitationCount", 0) if item else 0,
                    }
                    _ss_metrics_cache[pid] = metrics
                    result[pid] = metrics
                print(f"  📊 Semantic Scholar 获取成功：{len(batch)} 篇", file=sys.stderr)
            elif resp.status_code == 429:
                print("  ⚠️ Semantic Scholar 限流，等待后重试...", file=sys.stderr)
                time.sleep(5)
                # 重试一次
                resp2 = requests.post(url, params=params, json=body, timeout=20)
                if resp2.status_code == 200:
                    data = resp2.json()
                    for pid, item in zip(batch, data):
                        metrics = {
                            "citationCount": item.get("citationCount", 0) if item else 0,
                            "influentialCitationCount": item.get("influentialCitationCount", 0) if item else 0,
                        }
                        _ss_metrics_cache[pid] = metrics
                        result[pid] = metrics
                else:
                    for pid in batch:
                        _ss_metrics_cache[pid] = {"citationCount": 0, "influentialCitationCount": 0}
            else:
                print(f"  ⚠️ Semantic Scholar API 错误：{resp.status_code}", file=sys.stderr)
                for pid in batch:
                    _ss_metrics_cache[pid] = {"citationCount": 0, "influentialCitationCount": 0}
        except Exception as e:
            print(f"  ⚠️ Semantic Scholar 请求失败：{e}", file=sys.stderr)
            for pid in batch:
                _ss_metrics_cache[pid] = {"citationCount": 0, "influentialCitationCount": 0}
        time.sleep(1)  # 避免请求过快

    # 合并缓存中的结果
    all_result = {}
    for pid in [re.sub(r'v\d+$', '', p) for p in paper_ids]:
        if pid in _ss_metrics_cache:
            all_result[pid] = _ss_metrics_cache[pid]
    return all_result


def compute_hotness(paper: dict, hit_count: int, ss_metrics: dict = None,
                    hf_metrics: dict = None, arxiv_version: int = 1) -> float:
    """
    计算论文热度分数（越高越热门）。

    评分维度（满分 100 分）：
    - 引用次数 citationCount          30分：log 归一化，1000引用约得满分
    - 影响力引用 influentialCitations  15分：高质量引用信号
    - 时效性分数                       25分：越新越高，7天线性衰减
    - HF 点赞数 upvotes               20分：社区热度，对新论文最有效
    - arXiv 版本数                    5分：v2/v3 说明有反馈修订
    - 跨分类命中 hit_count             5分：被多个主题搜到

    ss_metrics: {"citationCount": int, "influentialCitationCount": int}，可为 None
    hf_metrics: {"upvotes": int, "comments": int}，可为 None
    arxiv_version: arXiv 版本号，默认 1
    """
    import math
    from datetime import date

    score = 0.0
    if ss_metrics is None:
        ss_metrics = {}
    if hf_metrics is None:
        hf_metrics = {}

    # ── 1. 引用次数（30分）──
    cite_count = ss_metrics.get("citationCount", 0) or 0
    # log 归一化：log2(cite+1) / log2(1001) * 30，1000引用约得满分
    cite_score = math.log2(cite_count + 1) / math.log2(1001) * 30.0
    score += min(cite_score, 30.0)

    # ── 2. 影响力引用（15分）──
    inf_cite = ss_metrics.get("influentialCitationCount", 0) or 0
    # log 归一化：log2(inf+1) / log2(51) * 15，50个影响力引用约得满分
    inf_score = math.log2(inf_cite + 1) / math.log2(51) * 15.0
    score += min(inf_score, 15.0)

    # ── 3. 时效性分数（25分）──
    try:
        pub = paper.get("published", "")  # 格式 MM-DD 或 YYYY-MM-DD
        today = date.today()
        parts = pub.split("-")
        if len(parts) == 3:
            pub_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            pub_date = date(today.year, int(parts[0]), int(parts[1]))
        else:
            pub_date = today
        days_ago = max((today - pub_date).days, 0)
        # 7天内线性衰减：第0天=25分，第7天=0分
        freshness = max(0.0, (7 - days_ago) / 7.0) * 25.0
        score += freshness
    except Exception:
        score += 8.0  # 无法解析日期给默认分

    # ── 4. HF 点赞数（20分）──
    # 对新论文最有效：发布当天即可获得社区反馈
    # log 归一化：log2(upvotes+1) / log2(101) * 20，100个点赞约得满分
    upvotes = hf_metrics.get("upvotes", 0) or 0
    hf_score = math.log2(upvotes + 1) / math.log2(101) * 20.0
    score += min(hf_score, 20.0)

    # ── 5. arXiv 版本数（5分）──
    # v1=0分，v2=3分，v3及以上=5分（有修订说明收到社区反馈）
    if arxiv_version >= 3:
        version_score = 5.0
    elif arxiv_version == 2:
        version_score = 3.0
    else:
        version_score = 0.0
    score += version_score

    # ── 6. 跨分类命中（5分）──
    # hit_count=1 得 2.5 分，hit_count>=2 得 5 分
    cross_score = min(hit_count * 2.5, 5.0)
    score += cross_score

    return round(score, 2)


def build_top10_table(all_papers: dict) -> str:
    """
    汇总所有分类论文，按热度排序，返回 Top 10 Markdown 表格。
    all_papers: {paper_id: (paper_dict, hit_count)}
    """
    # 批量获取 Semantic Scholar 引用数据
    all_ids = list(all_papers.keys())
    ss_data = fetch_semantic_scholar_metrics(all_ids)

    scored = []
    for pid, (paper, hit_count) in all_papers.items():
        clean_id = re.sub(r'v\d+$', '', pid)
        metrics = ss_data.get(clean_id, {})
        hf_metrics = fetch_hf_paper_metrics(clean_id)
        arxiv_ver = get_arxiv_version(pid)
        score = compute_hotness(paper, hit_count, ss_metrics=metrics,
                                hf_metrics=hf_metrics, arxiv_version=arxiv_ver)
        scored.append((score, paper, hit_count, metrics, hf_metrics, arxiv_ver))

    scored.sort(key=lambda x: x[0], reverse=True)
    top10 = scored[:10]

    if not top10:
        return "_暂无数据_"

    lines = []
    lines.append("| 排名 | 标题 | 中文摘要总结 | 论文关键字 | 发布者 | 热度分 | 发布日期 |")
    lines.append("|------|------|------------|----------|---------|--------|---------|")

    for rank, (score, paper, hit_count, metrics, hf_metrics, arxiv_ver) in enumerate(top10, 1):
        title = paper["title"].replace("|", "\\|")
        summary = format_summary(paper["summary"], max_len=400).replace("|", "\\|")
        kw_list = extract_keyphrases_en(paper["summary"], title=paper["title"], top_n=5)
        keywords = ", ".join(kw_list).replace("|", "\\|")
        affiliation = extract_affiliation(paper["summary"], paper["authors"]).replace("|", "\\|")
        published = paper["published"]
        url = paper["arxiv_url"]
        cite = metrics.get("citationCount", 0)
        upvotes = hf_metrics.get("upvotes", 0)
        hot_label = f"{score:.0f}分"
        if cite > 0:
            hot_label += f" 📚{cite}"
        if upvotes > 0:
            hot_label += f" 👍{upvotes}"
        if arxiv_ver >= 2:
            hot_label += f" v{arxiv_ver}"
        if hit_count > 1:
            hot_label += f" 🔥×{hit_count}"

        title_link = f"[{title}]({url})"
        lines.append(f"| {rank} | {title_link} | {summary} | {keywords} | {affiliation} | {hot_label} | {published} |")

    return "\n".join(lines)


def search_topic(topic: dict) -> list:
    """
    对单个主题执行搜索，返回论文列表。
    每个关键词单独搜索，结果合并去重，避免 AND 逻辑导致漏搜。
    """
    searcher = ArxivSearcher()
    seen_ids = set()
    all_results = []

    for kw in topic["keywords"]:
        try:
            results = searcher.search(
                keywords=[kw],
                days=topic["days"],
                max_results=topic["num"],
            )
            for paper in results:
                if paper["id"] not in seen_ids:
                    seen_ids.add(paper["id"])
                    all_results.append(paper)
        except Exception as e:
            print(f"⚠️ 搜索出错 [{topic['name']} / {kw}]: {e}", file=sys.stderr)

    # 按发布日期降序排序，截取 num 条
    all_results.sort(key=lambda p: p["published"], reverse=True)
    return all_results[:topic["num"]]


def build_table(papers: list) -> str:
    """将论文列表格式化为 Markdown 表格（含标题、摘要、机构、链接）"""
    if not papers:
        return "_本周期内暂无新论文_"

    lines = []
    lines.append("| # | 标题 | 内容摘要 | 发布机构 | 发布日期 | 链接 |")
    lines.append("|---|------|---------|---------|---------|------|")

    for i, paper in enumerate(papers, 1):
        title = paper["title"].replace("|", "\\|")
        summary = format_summary(paper["summary"], max_len=120).replace("|", "\\|")
        affiliation = extract_affiliation(paper["summary"], paper["authors"]).replace("|", "\\|")
        published = paper["published"]
        url = paper["arxiv_url"]
        short_id = paper["id"]

        lines.append(f"| {i} | {title} | {summary} | {affiliation} | {published} | [arxiv/{short_id}]({url}) |")

    return "\n".join(lines)


def build_echarts_wordcloud_data(all_papers_map: dict) -> list:
    """
    从所有论文的 PDF Keywords 字段提取关键词频率，返回 ECharts wordCloud 数据格式。
    同时把搜索所用的 TOPICS keywords 也纳入词云，并固定保留 vLLM / SGLang。
    [{name: "keyword", value: count}, ...]
    """
    word_freq: dict = {}

    # ① 把搜索所用的 TOPICS 关键词作为基础权重加入词云
    for topic in TOPICS:
        for kw in topic.get("keywords", []):
            kw = kw.strip()
            if kw:
                word_freq[kw] = word_freq.get(kw, 0) + 3  # 基础权重 3

    # ② 固定保留 vLLM 和 SGLang，确保始终出现在词云中
    for fixed_kw in ["vLLM", "SGLang"]:
        word_freq[fixed_kw] = word_freq.get(fixed_kw, 0) + 5  # 额外加权，保证可见

    # ③ 从论文内容中提取关键词并叠加权重
    for pid, (paper, hit_count) in all_papers_map.items():
        # 优先使用 HTML experimental 提取的 Keywords
        clean_id = re.sub(r'v\d+$', '', paper["id"])
        if clean_id in _html_cache:
            _, kws = _html_cache[clean_id]
        else:
            kws = []

        if not kws:
            # fallback：从摘要文本提取 Keywords 字段
            kws = extract_paper_keywords(paper["summary"])
        if not kws:
            # 最终 fallback：从 Abstract 提取英文名词短语（词云保持英文）
            abstract = paper["summary"]
            if clean_id in _html_cache:
                abstract, _ = _html_cache[clean_id]
            kws = extract_keyphrases_en(abstract, title=paper.get("title", ""), top_n=5)

        for kw in kws:
            weight = 1 + hit_count * 0.5
            word_freq[kw] = word_freq.get(kw, 0) + weight

    # 转换为 ECharts 格式，按频率降序，取 top 60
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:60]
    return [{"name": w, "value": round(v * 10)} for w, v in sorted_words]


def build_html_report(all_papers_map: dict, total_found: int) -> str:
    """生成美观的 HTML 格式报告，返回 HTML 字符串"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 批量获取 Semantic Scholar 引用数据
    all_ids = list(all_papers_map.keys())
    ss_data = fetch_semantic_scholar_metrics(all_ids)

    scored = []
    for pid, (paper, hit_count) in all_papers_map.items():
        clean_id = re.sub(r'v\d+$', '', pid)
        metrics = ss_data.get(clean_id, {})
        hf_metrics = fetch_hf_paper_metrics(clean_id)
        arxiv_ver = get_arxiv_version(pid)
        score = compute_hotness(paper, hit_count, ss_metrics=metrics,
                                hf_metrics=hf_metrics, arxiv_version=arxiv_ver)
        scored.append((score, paper, hit_count, metrics, hf_metrics, arxiv_ver))
    scored.sort(key=lambda x: x[0], reverse=True)
    top10 = scored[:10]

    # 热度分归一化到 0-100
    max_score = scored[0][0] if scored else 1.0
    min_score = scored[-1][0] if scored else 0.0
    score_range = max_score - min_score if max_score != min_score else 1.0

    def normalize_score(s):
        return round((s - min_score) / score_range * 100)

    rows_html = ""
    for rank, (score, paper, hit_count, metrics, hf_metrics, arxiv_ver) in enumerate(top10, 1):
        title = paper["title"].replace("<", "&lt;").replace(">", "&gt;")

        # 从 HTML experimental 提取 Abstract 和 Keywords
        pdf_abstract, pdf_keywords = get_paper_content_from_pdf(paper)

        # 中文摘要：对 PDF 提取的 Abstract 做总结翻译
        short_zh, _ = format_summary_dual(pdf_abstract)
        short_zh = short_zh.replace("<", "&lt;").replace(">", "&gt;")

        # hover 显示 PDF 原文 Abstract（英文）
        original_en_escaped = pdf_abstract.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

        # 关键词：优先来自 HTML Keywords 章节（英文原文），fallback 到 Abstract 提取英文关键短语
        if pdf_keywords:
            # HTML 提取的 Keywords 直接用英文，做格式清理
            paper_kws = []
            for kw in pdf_keywords[:6]:
                kw = kw.strip().strip('.,;·•-–—')
                if kw and 2 <= len(kw) <= 60:
                    paper_kws.append(kw)
        else:
            # 没有 Keywords 章节：从 Abstract + 标题提取英文关键短语
            paper_kws = extract_keyphrases_en(pdf_abstract, title=paper["title"], top_n=5)

        affiliation = extract_affiliation(pdf_abstract, paper["authors"]).replace("<", "&lt;").replace(">", "&gt;")
        published = paper["published"]
        url = paper["arxiv_url"]
        norm_score = normalize_score(score)
        fire_badge = f'<span class="fire-badge">🔥×{hit_count}</span>' if hit_count > 1 else ""

        # 引用数 badge
        cite_count = metrics.get("citationCount", 0) or 0
        inf_cite = metrics.get("influentialCitationCount", 0) or 0
        cite_badge = f'<span class="cite-badge" title="引用次数（来自 Semantic Scholar）">📚 {cite_count} 引用</span>' if cite_count > 0 else ""
        inf_badge = f'<span class="inf-cite-badge" title="高影响力引用次数">⭐ {inf_cite}</span>' if inf_cite > 0 else ""

        # HF 点赞数 badge
        upvotes = hf_metrics.get("upvotes", 0) or 0
        hf_badge = f'<span class="hf-badge" title="Hugging Face 社区点赞数">👍 {upvotes}</span>' if upvotes > 0 else ""

        # arXiv 版本 badge
        ver_badge = f'<span class="ver-badge" title="arXiv 版本（v2/v3 表示有修订）">v{arxiv_ver}</span>' if arxiv_ver >= 2 else ""

        # 关键词 badges
        kw_badges = "".join(f'<span class="kw-badge">{kw.strip()}</span>' for kw in paper_kws)

        # 热度进度条颜色
        bar_color = "#f97316" if norm_score >= 70 else ("#facc15" if norm_score >= 40 else "#4f8ef7")

        rows_html += f"""
        <tr class="paper-row">
            <td class="rank-cell">
                <span class="rank-num">#{rank}</span>
            </td>
            <td class="title-cell">
                <a href="{url}" target="_blank" class="paper-title">{title}</a>
            </td>
            <td class="summary-cell">
                <span class="summary-short" data-full="{original_en_escaped}">{short_zh}</span>
                <span class="summary-hint">🔍 hover查看英文原文</span>
            </td>
            <td class="kw-cell">{kw_badges}</td>
            <td class="affil-cell">{affiliation}</td>
            <td class="hot-cell">
                <span class="hot-score">{norm_score}</span>
                <div class="hot-bar-bg"><div class="hot-bar" style="width:{norm_score}%;background:{bar_color}"></div></div>
                <div class="hot-badges">{cite_badge}{inf_badge}{hf_badge}{ver_badge}{fire_badge}</div>
            </td>
            <td class="date-cell">{published}</td>
        </tr>"""

    # 生成 ECharts 词云数据
    import json as _json
    wc_data = build_echarts_wordcloud_data(all_papers_map)
    wc_data_json = _json.dumps(wc_data, ensure_ascii=False)

    # 生成搜索关键词按主题分组的 HTML
    search_kw_rows = ""
    for topic in TOPICS:
        topic_name = topic.get("name", "未知主题")
        kws = topic.get("keywords", [])
        kw_tags = "".join(f'<span class="search-kw-tag">{kw}</span>' for kw in kws)
        search_kw_rows += f'<div class="search-kw-row"><span class="search-kw-topic">{topic_name}</span><span class="search-kw-tags">{kw_tags}</span></div>\n'

    wordcloud_section = f"""
  <p class="section-title">🔍 论文搜索关键词（按主题分组）</p>
  <div class="search-kw-wrap">
    {search_kw_rows}
  </div>
  <p class="section-title">☁️ 论文关键词词云</p>
  <div class="wordcloud-wrap">
    <div id="wordcloud-chart" style="width:100%;height:380px;"></div>
  </div>
"""

    # 生成所有论文列表
    all_papers_rows = ""
    for i, (score, paper, hit_count, metrics, hf_metrics, arxiv_ver) in enumerate(scored, 1):
        t = paper["title"].replace("<", "&lt;").replace(">", "&gt;")
        url = paper["arxiv_url"]
        pub = paper["published"]
        ns = normalize_score(score)
        cite = metrics.get("citationCount", 0) or 0
        inf_cite = metrics.get("influentialCitationCount", 0) or 0
        upvotes = hf_metrics.get("upvotes", 0) or 0
        github_repo = hf_metrics.get("githubRepo", "") or ""
        github_stars = hf_metrics.get("githubStars", 0) or 0
        cite_td = f'<span style="color:#4f8ef7;font-weight:600">{cite}</span>' if cite > 0 else '<span style="color:#4a5568">—</span>'
        inf_td = f'<span style="color:#a78bfa;font-weight:600">{inf_cite}</span>' if inf_cite > 0 else '<span style="color:#4a5568">—</span>'
        hf_td = f'<span style="color:#34d399;font-weight:600">👍 {upvotes}</span>' if upvotes > 0 else '<span style="color:#4a5568">—</span>'
        ver_td = f'<span style="color:#facc15;font-weight:600">v{arxiv_ver}</span>' if arxiv_ver >= 2 else '<span style="color:#4a5568">v1</span>'
        if github_repo:
            stars_str = f' ⭐{github_stars}' if github_stars > 0 else ''
            code_td = f'<a href="{github_repo}" target="_blank" style="color:#4ade80;font-size:0.75rem;font-weight:600">✅ Code{stars_str}</a>'
        else:
            code_td = '<span style="color:#4a5568">—</span>'
        all_papers_rows += f'<tr><td>{i}</td><td><a href="{url}" target="_blank">{t}</a></td><td>{pub}</td><td>{ns}</td><td style="text-align:center">{cite_td}</td><td style="text-align:center">{inf_td}</td><td style="text-align:center">{hf_td}</td><td style="text-align:center">{ver_td}</td><td style="text-align:center">{code_td}</td></tr>\n'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Infra & 推理论文周报 · {now}</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0d0f14;
    --surface: #161a23;
    --surface2: #1e2330;
    --border: rgba(255,255,255,0.07);
    --accent: #4f8ef7;
    --accent2: #a78bfa;
    --hot: #f97316;
    --green: #34d399;
    --text: #e2e8f0;
    --text-muted: #64748b;
    --text-dim: #94a3b8;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Grotesk', system-ui, sans-serif;
    min-height: 100vh;
    padding: 40px 20px 80px;
  }}

  body::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
      radial-gradient(ellipse 80% 50% at 20% 10%, rgba(79,142,247,0.08) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 80%, rgba(167,139,250,0.06) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
  }}

  .container {{
    position: relative;
    z-index: 1;
    max-width: 1400px;
    margin: 0 auto;
  }}

  .header {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 40px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
  }}

  .header-left h1 {{
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.4rem, 3vw, 2rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.2;
    background: linear-gradient(135deg, #fff 30%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}

  .header-left .subtitle {{
    margin-top: 8px;
    font-size: 0.85rem;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}

  .header-right {{ text-align: right; }}

  .meta-tag {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text-dim);
    margin-left: 8px;
  }}

  .stats-bar {{
    display: flex;
    gap: 16px;
    margin-bottom: 32px;
    flex-wrap: wrap;
  }}

  .stat-card {{
    flex: 1;
    min-width: 140px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}

  .stat-card .stat-label {{
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  .stat-card .stat-value {{
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
  }}

  .section-title {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  .section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  .table-wrap {{
    overflow-x: auto;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--surface);
    margin-bottom: 40px;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }}

  thead tr {{
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
  }}

  thead th {{
    padding: 14px 16px;
    text-align: left;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    font-weight: 600;
    white-space: nowrap;
  }}

  .paper-row {{
    border-bottom: 1px solid var(--border);
    transition: background 0.2s;
  }}

  .paper-row:last-child {{ border-bottom: none; }}
  .paper-row:hover {{ background: rgba(79, 142, 247, 0.04); }}

  td {{
    padding: 14px 16px;
    vertical-align: top;
  }}

  .rank-cell {{
    width: 52px;
    text-align: center;
    padding-top: 16px;
  }}

  .rank-num {{
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-muted);
  }}

  .paper-row:nth-child(1) .rank-num {{ color: #fbbf24; font-size: 1rem; }}
  .paper-row:nth-child(2) .rank-num {{ color: #94a3b8; font-size: 0.95rem; }}
  .paper-row:nth-child(3) .rank-num {{ color: #cd7c4a; font-size: 0.9rem; }}

  .title-cell {{
    min-width: 240px;
    max-width: 300px;
  }}

  .paper-title {{
    color: var(--text);
    text-decoration: none;
    font-weight: 500;
    line-height: 1.4;
    display: block;
    transition: color 0.2s;
  }}

  .paper-title:hover {{
    color: var(--accent);
    text-decoration: underline;
    text-underline-offset: 3px;
  }}

  /* 摘要 hover tooltip */
  .summary-cell {{
    min-width: 220px;
    max-width: 300px;
    color: var(--text-dim);
    font-size: 0.83rem;
    line-height: 1.7;
    position: relative;
  }}

  .summary-short {{
    cursor: help;
    border-bottom: 1px dashed rgba(100,116,139,0.4);
    display: inline;
  }}

  .summary-hint {{
    display: block;
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 4px;
    opacity: 0.7;
  }}

  /* 自定义 tooltip 浮层 */
  #summary-tooltip {{
    position: fixed;
    z-index: 9999;
    max-width: 420px;
    background: #1e2330;
    border: 1px solid rgba(79,142,247,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: var(--text-dim);
    line-height: 1.7;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    pointer-events: none;
    display: none;
    word-break: break-word;
  }}

  /* 搜索关键词展示区域 */
  .search-kw-wrap {{
    border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--surface);
    padding: 16px 20px;
    margin-bottom: 28px;
  }}
  .search-kw-row {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }}
  .search-kw-row:last-child {{
    border-bottom: none;
  }}
  .search-kw-topic {{
    flex-shrink: 0;
    min-width: 120px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--accent);
    padding-top: 3px;
  }}
  .search-kw-tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }}
  .search-kw-tag {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.3);
    color: #a0c4ff;
    white-space: nowrap;
  }}

  /* 词云 */
  .wordcloud-wrap {{
    border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--surface);
    overflow: hidden;
    margin-bottom: 40px;
    padding: 8px;
  }}

  .kw-cell {{
    min-width: 160px;
    max-width: 200px;
  }}

  .kw-badge {{
    display: inline-block;
    padding: 2px 8px;
    margin: 2px 3px 2px 0;
    border-radius: 6px;
    font-size: 0.72rem;
    background: rgba(79,142,247,0.1);
    border: 1px solid rgba(79,142,247,0.2);
    color: var(--accent);
    white-space: nowrap;
  }}

  .affil-cell {{
    min-width: 100px;
    color: var(--text-dim);
    font-size: 0.82rem;
  }}

  .hot-cell {{
    white-space: nowrap;
    text-align: center;
    min-width: 110px;
  }}

  .hot-score {{
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--hot);
  }}

  .hot-bar-bg {{
    width: 70px;
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    margin: 4px auto 2px;
    overflow: hidden;
  }}

  .hot-bar {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
  }}

  .hot-badges {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 3px;
    margin-top: 3px;
  }}

  .fire-badge {{
    display: inline-block;
    font-size: 0.68rem;
    color: var(--hot);
  }}

  .cite-badge {{
    display: inline-block;
    font-size: 0.66rem;
    color: #60a5fa;
    background: rgba(96,165,250,0.12);
    border-radius: 4px;
    padding: 1px 5px;
    cursor: default;
  }}

  .inf-cite-badge {{
    display: inline-block;
    font-size: 0.66rem;
    color: #fbbf24;
    background: rgba(251,191,36,0.12);
    border-radius: 4px;
    padding: 1px 5px;
    cursor: default;
  }}

  .hf-badge {{
    display: inline-block;
    font-size: 0.66rem;
    color: #34d399;
    background: rgba(52,211,153,0.12);
    border-radius: 4px;
    padding: 1px 5px;
    cursor: default;
  }}

  .ver-badge {{
    display: inline-block;
    font-size: 0.64rem;
    color: #facc15;
    background: rgba(250,204,21,0.1);
    border-radius: 4px;
    padding: 1px 5px;
    cursor: default;
    font-family: 'Space Mono', monospace;
  }}

  .date-cell {{
    white-space: nowrap;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    min-width: 80px;
  }}

  /* 搜索关键词展示区域 */
  .search-kw-wrap {{
    border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--surface);
    padding: 16px 20px;
    margin-bottom: 28px;
  }}
  .search-kw-row {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }}
  .search-kw-row:last-child {{
    border-bottom: none;
  }}
  .search-kw-topic {{
    flex-shrink: 0;
    min-width: 120px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--accent);
    padding-top: 3px;
  }}
  .search-kw-tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }}
  .search-kw-tag {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.3);
    color: #a0c4ff;
    white-space: nowrap;
  }}

  /* 词云 */
  .wordcloud-wrap {{
    border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--surface);
    overflow: hidden;
    margin-bottom: 40px;
    text-align: center;
    padding: 8px;
  }}

  .kw-cell {{
    min-width: 160px;
    max-width: 200px;
  }}
  .all-papers-wrap {{
    border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--surface);
    overflow: hidden;
    margin-bottom: 40px;
  }}

  .all-papers-wrap summary {{
    padding: 14px 20px;
    cursor: pointer;
    font-size: 0.82rem;
    color: var(--text-dim);
    background: var(--surface2);
    user-select: none;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .all-papers-wrap summary::-webkit-details-marker {{ display: none; }}

  .all-papers-wrap summary::before {{
    content: '▶';
    font-size: 0.65rem;
    transition: transform 0.2s;
  }}

  .all-papers-wrap[open] summary::before {{
    transform: rotate(90deg);
  }}

  .all-papers-wrap summary:hover {{ color: var(--text); }}

  .all-papers-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }}

  .all-papers-table th {{
    padding: 10px 16px;
    text-align: left;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
  }}

  .all-papers-table td {{
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}

  .all-papers-table tr:last-child td {{ border-bottom: none; }}
  .all-papers-table tr:hover td {{ background: rgba(79,142,247,0.03); }}

  .all-papers-table a {{
    color: var(--text-dim);
    text-decoration: none;
    transition: color 0.2s;
  }}

  .all-papers-table a:hover {{
    color: var(--accent);
    text-decoration: underline;
  }}

  .formula-section {{
    margin-top: 40px;
  }}

  .formula-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 16px;
  }}

  .formula-box {{
    background: rgba(79,142,247,0.06);
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 20px;
  }}

  .formula-title {{
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  .formula-expr {{
    font-size: 0.95rem;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    line-height: 1.6;
  }}

  .formula-note {{
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 6px;
  }}

  .formula-dims {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
  }}

  .formula-dim {{
    background: var(--surface2);
    border-radius: 8px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}

  .dim-name {{
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text);
  }}

  .dim-weight {{
    font-size: 0.82rem;
    color: var(--accent2);
    font-family: 'Space Mono', monospace;
  }}

  .dim-desc {{
    font-size: 0.78rem;
    color: var(--text-dim);
    line-height: 1.5;
    margin-top: 2px;
  }}

  .formula-sources {{
    font-size: 0.78rem;
    color: var(--text-muted);
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }}

  .formula-sources a {{
    color: var(--accent);
    text-decoration: none;
  }}

  .footer {{
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    color: var(--text-muted);
    flex-wrap: wrap;
    gap: 8px;
  }}

  .footer a {{ color: var(--accent); text-decoration: none; }}

  @media (max-width: 768px) {{
    .header {{ flex-direction: column; align-items: flex-start; gap: 12px; }}
    .summary-cell, .kw-cell {{ display: none; }}
  }}
</style>
</head>
<body>
<div id="summary-tooltip"></div>
<div class="container">
    <div class="header-left">
      <h1>📚 AI Infra &amp; 推理论文周报</h1>
      <p class="subtitle">LLM Inference · KV Cache · AI Infrastructure · Speculative Decoding · vLLM · SGLang · DeepSeek · Qwen</p>
    </div>
    <div class="header-right">
      <span class="meta-tag">📅 {now}</span>
      <span class="meta-tag">⏱ 最近7天</span>
    </div>
  </header>

  <div class="stats-bar">
    <div class="stat-card">
      <span class="stat-label">发现论文</span>
      <span class="stat-value">{total_found}</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">精选 Top</span>
      <span class="stat-value">{len(top10)}</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">监控主题</span>
      <span class="stat-value">{len(TOPICS)}</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">热度最高</span>
      <span class="stat-value" style="font-size:1.1rem;color:var(--hot)">100分</span>
    </div>
  </div>

  <p class="section-title">🏆 本周热门论文 Top 10 — 综合热度排名</p>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>排名</th>
          <th>标题</th>
          <th>中文摘要</th>
          <th>关键词</th>
          <th>发布者</th>
          <th>热度分 (0-100)</th>
          <th>发布日期</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  {wordcloud_section}

  <p class="section-title">📋 所有采集论文列表</p>
  <details class="all-papers-wrap">
    <summary>点击展开 · 共 {len(scored)} 篇论文</summary>
    <table class="all-papers-table">
      <thead>
        <tr>
          <th>#</th>
          <th>标题</th>
          <th>发布日期</th>
          <th>热度分</th>
          <th>引用次数</th>
          <th>影响力引用</th>
          <th>HF 点赞</th>
          <th>arXiv 版本</th>
          <th>GitHub 代码</th>
        </tr>
      </thead>
      <tbody>
        {all_papers_rows}
      </tbody>
    </table>
  </details>

  <section class="formula-section">
    <p class="section-title">📐 热度评分公式详解</p>
    <div class="formula-wrap">
      <div class="formula-box">
        <div class="formula-title">综合热度分（满分 100 分）</div>
        <div class="formula-expr">
          热度分 = 引用次数分（30）+ 影响力引用分（15）+ 时效性分（25）+ HF点赞分（20）+ 版本分（5）+ 跨分类命中分（5）
        </div>
        <div class="formula-note">最终归一化至 0–100 分</div>
      </div>
      <div class="formula-dims">
        <div class="formula-dim">
          <span class="dim-name">📚 引用次数</span>
          <span class="dim-weight">30 分</span>
          <span class="dim-desc">来自 Semantic Scholar。log₂(引用数+1) / log₂(1001) × 30，约 1000 次引用得满分。对成熟论文最有效。</span>
        </div>
        <div class="formula-dim">
          <span class="dim-name">⭐ 影响力引用</span>
          <span class="dim-weight">15 分</span>
          <span class="dim-desc">来自 Semantic Scholar influentialCitationCount。log₂(引用数+1) / log₂(51) × 15，约 50 次高影响力引用得满分。衡量论文质量而非数量。</span>
        </div>
        <div class="formula-dim">
          <span class="dim-name">⏰ 时效性</span>
          <span class="dim-weight">25 分</span>
          <span class="dim-desc">发布当天得 25 分，7 天内线性衰减，超过 7 天得 0 分。确保新论文不会因无引用而被埋没。</span>
        </div>
        <div class="formula-dim">
          <span class="dim-name">👍 HF 点赞数</span>
          <span class="dim-weight">20 分</span>
          <span class="dim-desc">来自 Hugging Face Papers 社区点赞数。log₂(点赞数+1) / log₂(101) × 20，约 100 个点赞得满分。<strong>对新论文最有效</strong>，发布当天即可获得社区反馈。</span>
        </div>
        <div class="formula-dim">
          <span class="dim-name">🔄 arXiv 版本数</span>
          <span class="dim-weight">5 分</span>
          <span class="dim-desc">v1=0分，v2=3分，v3及以上=5分。有修订版本说明论文收到了社区反馈，作者在积极完善。</span>
        </div>
        <div class="formula-dim">
          <span class="dim-name">🔥 跨分类命中</span>
          <span class="dim-weight">5 分</span>
          <span class="dim-desc">被多个搜索主题命中：命中1个主题得2.5分，命中2个及以上得5分。说明论文覆盖多个研究方向。</span>
        </div>
      </div>
      <div class="formula-sources">
        数据来源：
        <a href="https://www.semanticscholar.org" target="_blank">Semantic Scholar</a>（引用数）·
        <a href="https://huggingface.co/papers" target="_blank">Hugging Face Papers</a>（社区热度）·
        <a href="https://arxiv.org" target="_blank">arXiv.org</a>（版本信息）
      </div>
    </div>
  </section>

  <footer class="footer">
    <span>热度评分 = 引用次数（30%）+ 影响力引用（15%）+ 时效性（25%）+ HF点赞（20%）+ arXiv版本（5%）+ 跨分类命中（5%）</span>
    <span>数据来源: <a href="https://arxiv.org" target="_blank">arXiv.org</a></span>
  </footer>
</div>

<!-- ECharts + wordcloud -->
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts-wordcloud@2/dist/echarts-wordcloud.min.js"></script>
<script>
(function() {{
  // ---- ECharts 词云 ----
  var chartDom = document.getElementById('wordcloud-chart');
  if (chartDom) {{
    var myChart = echarts.init(chartDom, null, {{renderer: 'canvas'}});
    var wcData = {wc_data_json};
    myChart.setOption({{
      backgroundColor: '#161a23',
      tooltip: {{
        show: true,
        formatter: function(p) {{ return p.name + ': ' + p.value; }}
      }},
      series: [{{
        type: 'wordCloud',
        shape: 'pentagon',
        left: 'center',
        top: 'center',
        width: '95%',
        height: '90%',
        sizeRange: [14, 72],
        rotationRange: [-45, 45],
        rotationStep: 15,
        gridSize: 8,
        drawOutOfBound: false,
        layoutAnimation: true,
        textStyle: {{
          fontFamily: 'Space Grotesk, sans-serif',
          fontWeight: 'bold',
          color: function() {{
            var colors = [
              '#4f8ef7','#a78bfa','#34d399','#f97316',
              '#facc15','#38bdf8','#fb7185','#6ee7b7'
            ];
            return colors[Math.floor(Math.random() * colors.length)];
          }}
        }},
        emphasis: {{
          focus: 'self',
          textStyle: {{ shadowBlur: 10, shadowColor: '#4f8ef7' }}
        }},
        data: wcData
      }}]
    }});
    window.addEventListener('resize', function() {{ myChart.resize(); }});
  }}

  // ---- Summary hover tooltip ----
  var tooltip = document.getElementById('summary-tooltip');
  var allShorts = document.querySelectorAll('.summary-short');
  allShorts.forEach(function(el) {{
    el.addEventListener('mouseenter', function(e) {{
      var full = el.getAttribute('data-full');
      if (!full) return;
      tooltip.textContent = full;
      tooltip.style.display = 'block';
      positionTooltip(e);
    }});
    el.addEventListener('mousemove', function(e) {{
      positionTooltip(e);
    }});
    el.addEventListener('mouseleave', function() {{
      tooltip.style.display = 'none';
    }});
  }});

  function positionTooltip(e) {{
    var x = e.clientX + 16;
    var y = e.clientY + 16;
    var tw = tooltip.offsetWidth || 420;
    var th = tooltip.offsetHeight || 100;
    if (x + tw > window.innerWidth - 10) x = e.clientX - tw - 10;
    if (y + th > window.innerHeight - 10) y = e.clientY - th - 10;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }}
}})();
</script>
</body>
</html>"""
    return html

def run_monitor():
    """执行所有主题监控，返回汇总报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# 📚 AI Infra & 推理论文周报")
    lines.append(f"**生成时间**: {now}  |  **监控周期**: 最近7天")
    lines.append("")

    total_found = 0
    # 汇总所有论文，用于 Top 10 排名：{paper_id: (paper, hit_count)}
    all_papers_map: dict = {}

    for topic in TOPICS:
        papers = search_topic(topic)
        total_found += len(papers)
        for paper in papers:
            pid = paper["id"]
            if pid in all_papers_map:
                all_papers_map[pid] = (all_papers_map[pid][0], all_papers_map[pid][1] + 1)
            else:
                all_papers_map[pid] = (paper, 1)

    # 只输出 Top 10 热门论文榜
    lines.append("## 🏆 本周热门论文 Top 10")
    lines.append("> 热度评分综合考虑：引用次数 📚（30%）、影响力引用 ⭐（15%）、发布时效性 ⏰（25%）、HF点赞 👍（20%）、arXiv版本 🔄（5%）、跨分类命中 🔥（5%）；数据来源 Semantic Scholar & Hugging Face Papers")
    lines.append("")
    lines.append(build_top10_table(all_papers_map))
    lines.append("")

    lines.append("---")
    lines.append(f"**✅ 本次监控完成，共发现约 {total_found} 篇相关论文，已筛选热度 Top 10**")

    # 生成 HTML 文件
    html_content = build_html_report(all_papers_map, total_found)
    date_str = datetime.now().strftime('%Y%m%d_%H%M')
    html_filename = f"/data/workspace/paper_report_{date_str}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    # 推送到 GitHub 仓库
    github_url = push_to_github(html_filename, html_content, date_str)

    report_text = "\n".join(lines)
    return report_text, html_filename, github_url


def push_to_github(html_filename: str, html_content: str, date_str: str) -> str:
    """
    将 HTML 报告推送到 GitHub 仓库 feiqiangs/ai_paper_summary。
    同时更新 index.html（最新报告）和按日期命名的归档文件。
    返回可访问的 GitHub Pages 链接。
    """
    import subprocess
    import shutil

    repo_dir = "/data/workspace/ai_paper_summary"
    today = datetime.now().strftime('%Y-%m-%d')

    try:
        # 确保仓库存在并更新
        if not os.path.exists(repo_dir):
            subprocess.run(
                ["git", "clone", "git@github.com:feiqiangs/ai_paper_summary.git", repo_dir],
                check=True, capture_output=True
            )
        else:
            subprocess.run(["git", "pull", "--rebase"], cwd=repo_dir, capture_output=True)

        # 写入 index.html（最新报告，始终覆盖）
        index_path = os.path.join(repo_dir, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 写入按日期归档的文件
        archive_name = f"report_{date_str}.html"
        archive_path = os.path.join(repo_dir, archive_name)
        shutil.copy(html_filename, archive_path)

        # Git commit & push
        subprocess.run(["git", "add", "index.html", archive_name], cwd=repo_dir, check=True, capture_output=True)
        commit_msg = f"📚 AI论文周报 {today} ({date_str})"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_dir, check=True, capture_output=True)
        result = subprocess.run(["git", "push", "origin", "main"], cwd=repo_dir, capture_output=True)

        if result.returncode != 0:
            # 尝试 master 分支
            result2 = subprocess.run(["git", "push", "origin", "master"], cwd=repo_dir, capture_output=True)
            if result2.returncode != 0:
                return f"⚠️ Push 失败: {result2.stderr.decode()}"

        # 返回 GitHub Pages 完整链接（带具体文件名）
        pages_url = f"https://feiqiangs.github.io/ai_paper_summary/report_{date_str}.html"
        return pages_url

    except subprocess.CalledProcessError as e:
        return f"⚠️ GitHub 操作失败: {e.stderr.decode() if e.stderr else str(e)}"
    except Exception as e:
        return f"⚠️ 推送异常: {str(e)}"


if __name__ == "__main__":
    report, html_path, github_url = run_monitor()
    print(report)
    print(f"\n📄 HTML报告已生成: {html_path}")
    print(f"🔗 GitHub Pages 链接: {github_url}")