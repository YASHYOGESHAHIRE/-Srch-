import os
import re
import json
import time
import hashlib
import urllib.request
import urllib.error
from html.parser import HTMLParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from groq import Groq
from ddgs import DDGS
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# CONFIG
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

app = FastAPI(title="Perplexity Clone API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS — Groq chat completion wrapper
# ============================================================
def _chat(system: str, user: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


# ============================================================
# REQUEST / RESPONSE MODELS — Search
# ============================================================
class SearchRequest(BaseModel):
    query: str
    history: List[str] = []

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    followups: List[str]
    elapsed: float


# ============================================================
# STAGE 1: WEB SEARCH
# ============================================================
def search_web(query: str, max_results: int = 10) -> list:
    try:
        results = DDGS().text(query=query, max_results=max_results)
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
                "domain":  re.sub(r"https?://(www\.)?", "", r.get("href", "")).split("/")[0],
            }
            for r in results if r.get("href")
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []


# ============================================================
# STAGE 2: PARALLEL SCRAPING
# ============================================================
class _TextExtractor(HTMLParser):
    _SKIP_TAGS = {"script","style","nav","footer","header","aside","form","iframe"}
    def __init__(self):
        super().__init__()
        self._depth = 0
        self.parts: list[str] = []
    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS: self._depth += 1
    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._depth: self._depth -= 1
    def handle_data(self, data):
        if not self._depth:
            s = data.strip()
            if s: self.parts.append(s)

def scrape_url(url: str, timeout: int = 8) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        e = _TextExtractor(); e.feed(html)
        return re.sub(r"\s+", " ", " ".join(e.parts))[:8000]
    except Exception:
        return ""

def scrape_sources_parallel(sources: list, top_n: int = 5) -> list:
    selected = sources[:top_n]
    with ThreadPoolExecutor(max_workers=top_n) as executor:
        futures = {executor.submit(scrape_url, s["url"]): s for s in selected}
        for f in as_completed(futures):
            futures[f]["content"] = f.result()
    return [s for s in selected if len(s.get("content","")) > 300]


# ============================================================
# STAGE 3: LIGHTWEIGHT RETRIEVAL
# ============================================================
def _chunk_text(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size-overlap) if len(text[i:i+size]) > 100]

def _keyword_score(query: str, chunk: str) -> float:
    q = set(re.findall(r"\w+", query.lower()))
    c = Counter(re.findall(r"\w+", chunk.lower()))
    return sum(c[t] for t in q)

def retrieve_context(query: str, sources: list, top_k: int = 10) -> tuple[str, list]:
    chunks, meta = [], []
    for src in sources:
        for ch in _chunk_text(src.get("content","")):
            chunks.append(ch); meta.append(src)
    if not chunks: return "", []
    scored = sorted(range(len(chunks)), key=lambda i: _keyword_score(query, chunks[i]), reverse=True)[:top_k]
    parts, used, seen = [], [], set()
    for idx in scored:
        src = meta[idx]
        parts.append(f"[Source: {src['domain']}]\n{chunks[idx]}")
        if src["url"] not in seen:
            used.append(src); seen.add(src["url"])
    return "\n\n---\n\n".join(parts), used


# ============================================================
# STAGE 4: GENERATE ANSWER
# ============================================================
def generate_answer(query: str, context: str, search_results: list) -> str:
    source_list = "\n".join(f"[{i+1}] {r['url']} — {r['title']}" for i,r in enumerate(search_results[:8]))
    system = "You are a Perplexity AI-style research assistant. Give accurate, comprehensive answers with inline citations."
    user = f"""User Question: {query}

Retrieved Web Context:
{context}

Source List:
{source_list}

Instructions:
1. Write a detailed Markdown answer
2. Use inline citations like [1], [2], [3]
3. Use ## headers for multi-part answers
4. Cite every key claim
5. Use bullet points where helpful
6. Note conflicting info across sources if present
7. End with a ## Sources section

Answer:"""
    return _chat(system, user, max_tokens=2048)


# ============================================================
# STAGE 5: FOLLOW-UPS
# ============================================================
def generate_followups(query: str, answer: str) -> list:
    system = "Return ONLY a JSON array of 4 strings — no markdown, no preamble."
    user = f"Generate 4 natural follow-up questions.\n\nQuery: {query}\nAnswer: {answer[:500]}"
    try:
        raw = re.sub(r"```json|```", "", _chat(system, user, max_tokens=256, temperature=0.8)).strip()
        parsed = json.loads(raw)
        return parsed[:4] if isinstance(parsed, list) else []
    except Exception:
        return []


# ============================================================
# SEARCH PIPELINE
# ============================================================
def _enhance_query(query: str, history: list) -> str:
    triggers = {"it","that","this","they","he","she","more","why","how so","explain"}
    if history and query.lower().split()[0] in triggers:
        return f"{query} (follow-up to: '{history[-1]}')"
    return query

def run_pipeline(query: str, history: list) -> dict:
    start = time.time()
    q = _enhance_query(query, history)
    results = search_web(q)
    if not results: raise HTTPException(status_code=404, detail="No results found.")
    scraped = scrape_sources_parallel(results)
    context, used = retrieve_context(q, scraped)
    if not context:
        context = "\n\n".join(r["snippet"] for r in results[:5])
        used = results[:5]
    answer = generate_answer(q, context, results)
    followups = generate_followups(q, answer)
    return {"query": query, "answer": answer, "sources": used or results[:5],
            "followups": followups, "elapsed": round(time.time()-start, 2)}


# ============================================================
# DESIGN GENERATION
# ============================================================
_design_cache: dict = {}

class DesignRequest(BaseModel):
    prompt: str

class DesignItem(BaseModel):
    style: str
    title: str
    html: str

class DesignResponse(BaseModel):
    designs: List[DesignItem]
    cached: bool
    elapsed: float

_DIAGRAM_SYSTEM = """\
You are a world-class information designer, diagram architect, and visual storyteller.
Generate a single, complete, self-contained HTML document that creates a visually stunning,
production-quality diagram or visual information artifact from scratch.

OUTPUT RULES:
1. Return exactly one raw HTML document starting with <!DOCTYPE html>
2. Include all HTML, CSS, and optional vanilla JavaScript in one file
3. No React, no Tailwind, no Bootstrap, no Mermaid, no external libraries
4. You may use inline SVG authored manually
5. No markdown fences, no explanations, no preamble

DESIGN RULES:
1. Bold, intentional aesthetic matched to subject matter
2. Always @import Google Fonts with distinctive pairings — never Inter/Roboto/Arial
3. Strong color system with CSS custom properties
4. Never plain flat background — use gradients, textures, grids, depth treatments
5. Sophisticated composition: asymmetry, overlap, labeled zones, connector paths, legends
6. CSS animations for page load with staggered reveal timing
7. Hover micro-interactions on nodes, cards, labels
"""

_VARIANT_HINTS = {
    "dark":  "Dark, cinematic, high-contrast. Think premium blueprint, control room, neon schematic.",
    "light": "Light, refined, editorial. Think premium infographic, design annual, strategy wall.",
}

def _generate_html(prompt: str, variant: str) -> str:
    hint = _VARIANT_HINTS.get(variant, "")
    user = (f'Create a stunning HTML diagram for: "{prompt}"\n\nVisual direction: {hint}\n\n'
            "Output raw HTML only, starting with <!DOCTYPE html>.")
    raw = _chat(_DIAGRAM_SYSTEM, user, max_tokens=4096, temperature=0.92)
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    return re.sub(r"\n?```$", "", raw.strip(), flags=re.MULTILINE).strip()

def _extract_title(html: str, fallback: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE|re.DOTALL)
    return m.group(1).strip() if m else fallback


# ============================================================
# LEARN / TUTOR
# ============================================================
_learn_cache: dict = {}

_LEARN_SYSTEM = """\
You are an expert educator and curriculum designer. Teach any topic in a structured,
engaging, visual-friendly way.

Return ONLY a valid JSON object — no markdown fences, no explanation, no preamble.

Schema:
{
  "title": "Topic title",
  "emoji": "single relevant emoji",
  "tagline": "one punchy line that sells why this topic matters (max 12 words)",
  "sections": [
    {
      "type": "intro",
      "heading": "What is it?",
      "body": "2-3 sentence plain-English explanation. No jargon."
    },
    {
      "type": "visual",
      "heading": "How it works",
      "visual_type": "steps_flow | comparison | timeline | cycle | tree | network",
      "visual_data": {
        "nodes": [
          {"id": "A", "label": "Node label", "sublabel": "optional detail", "color": "blue|green|amber|red|purple|teal"}
        ],
        "edges": [
          {"from": "A", "to": "B", "label": "optional edge label"}
        ]
      }
    },
    {
      "type": "breakdown",
      "heading": "Key concepts",
      "items": [
        {"term": "Term", "definition": "One clear sentence.", "icon": "emoji"}
      ]
    },
    {
      "type": "analogy",
      "heading": "Think of it like this",
      "analogy_text": "A vivid real-world analogy in 2-3 sentences."
    },
    {
      "type": "code",
      "heading": "See it in code",
      "language": "python or javascript",
      "code": "concise working code example max 20 lines",
      "explanation": "one sentence explaining what the code shows"
    },
    {
      "type": "comparison",
      "heading": "X vs Y",
      "left_label": "Option A",
      "right_label": "Option B",
      "rows": [
        {"aspect": "Speed", "left": "description", "right": "description"},
        {"aspect": "Use case", "left": "description", "right": "description"},
        {"aspect": "Complexity", "left": "description", "right": "description"},
        {"aspect": "Best for", "left": "description", "right": "description"}
      ]
    },
    {
      "type": "quiz",
      "heading": "Test yourself",
      "questions": [
        {
          "q": "Question?",
          "options": ["A", "B", "C", "D"],
          "answer": 0,
          "explanation": "Why correct."
        }
      ]
    }
  ]
}

RULES:
- Section order: intro → visual → breakdown → analogy → code (skip if non-technical) → comparison (skip if no natural A vs B) → quiz
- visual_data.nodes: 4-7 nodes max. Only reference node ids that exist in edges.
- steps_flow: chain A→B→C. cycle: last node back to first. tree: one root branches out.
- Skip code section entirely for non-technical topics (history, biology basics, art, etc.)
- Skip comparison section if topic has no natural A vs B
- All text must be specific to the topic — no filler
"""

class LearnRequest(BaseModel):
    topic: str

class LearnResponse(BaseModel):
    topic: str
    data: Dict[str, Any]
    cached: bool
    elapsed: float


# ============================================================
# API ROUTES
# ============================================================
@app.get("/")
def root():
    return {"message": "Perplexity Clone API is running"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return run_pipeline(req.query.strip(), req.history)


@app.post("/generate-designs", response_model=DesignResponse)
def generate_designs(req: DesignRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    start = time.time()
    prompt = req.prompt.strip()
    cache_key = hashlib.md5(prompt.lower().encode()).hexdigest()
    if cache_key in _design_cache:
        return DesignResponse(designs=_design_cache[cache_key], cached=True,
                              elapsed=round(time.time()-start, 2))
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fd = ex.submit(_generate_html, prompt, "dark")
            fl = ex.submit(_generate_html, prompt, "light")
            html_dark, html_light = fd.result(), fl.result()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    designs = [
        DesignItem(style="dark",  title=_extract_title(html_dark,  f"{prompt} — Dark"),  html=html_dark),
        DesignItem(style="light", title=_extract_title(html_light, f"{prompt} — Light"), html=html_light),
    ]
    _design_cache[cache_key] = designs
    return DesignResponse(designs=designs, cached=False, elapsed=round(time.time()-start, 2))


@app.post("/learn", response_model=LearnResponse)
def learn(req: LearnRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    start = time.time()
    topic = req.topic.strip()
    cache_key = hashlib.md5(topic.lower().encode()).hexdigest()
    if cache_key in _learn_cache:
        return LearnResponse(topic=topic, data=_learn_cache[cache_key],
                             cached=True, elapsed=round(time.time()-start, 2))
    try:
        raw = _chat(_LEARN_SYSTEM, f"Teach me about: {topic}", max_tokens=4096, temperature=0.7)
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learn generation failed: {e}")
    _learn_cache[cache_key] = data
    return LearnResponse(topic=topic, data=data, cached=False, elapsed=round(time.time()-start, 2))