"""레시피 Q&A 챗봇 핵심 로직: 데이터 로드 → 검색(retrieve) → 프롬프트 구성 → LLM 호출."""
import csv
import hashlib
import json
import os
import re
import time

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("GB_BASE_URL", "https://api.gbcloud.ai/v1")  # check_api.py로 꼭 확인
API_KEY = os.getenv("GB_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "")  # 비워두면 키워드 검색만 사용

client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "missing")

SYSTEM_PROMPT = """너는 집밥 레시피 도우미야. 아래 [레시피 목록]에 있는 정보만으로 한국어로 답해.
규칙:
# - 목록에 없는 메뉴나 재료는 지어내지 말고 "레시피 목록에 없는 내용이에요"라고 말해.
- 질문한 메뉴가 목록에 있으면 목록 내용을 기준으로 답해.
- 목록에 없는 메뉴면 첫 줄에 "📌 레시피 목록에 없는 메뉴라 일반적인 방법으로 안내드려요."라고 밝힌 뒤, 일반적인 조리법을 알려줘.
- 재료명은 목록에 적힌 표기 그대로 써.
- 조리 방법을 물으면 [조리순서]를 단계별 번호로 정리해서 알려줘.
- 시간·난이도 조건(예: 15분 이내, 쉬운 것)이 있으면 그 기준으로 골라줘.
- "~없이", "~빼고" 같은 조건이 있으면 그 재료가 들어간 메뉴는 제외해.
- 메뉴를 추천할 때는 메뉴명과 필요한 재료를 함께 짧게 알려줘.
"""

MAX_HISTORY = 6  # 최근 대화 몇 개까지 같이 보낼지 (토큰 절약)


# ---------- 데이터 ----------
def load_recipes(path="recipes.csv"):
    """CSV 각 행을 레시피 dict로. 열 이름이 바뀌어도 'text'에 전부 들어가도록 일반화."""
    recipes = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            base, sauce = row.get("기본재료", ""), row.get("양념재료", "")
            ings = [s.strip() for s in f"{base}, {sauce}".split(",") if s.strip() not in ("", "-")]
            recipes.append({
                "menu": row["메뉴"],
                "ingredients": ings,
                "text": " | ".join(f"{k}: {v}" for k, v in row.items() if v),
                **row,
            })
    return recipes


def format_context(recipes):
    return "\n".join(f"- {r['text']}" for r in recipes)


# ---------- 검색 1: 키워드 ----------
def keyword_scores(question, recipes):
    q = question.replace(" ", "")
    scores = []
    for r in recipes:
        s = 0.0
        for name in r["menu"].split("/"):
            if name.replace(" ", "") in q:
                s += 3
        for ing in r["ingredients"]:
            name = ing.replace(" ", "")
            if len(name) >= 2 and name in q:
                s += 1
        scores.append(s)
    return np.array(scores)


# ---------- 검색 2: 임베딩 (EMBED_MODEL 설정 시) ----------
def _embed(texts, batch=32):
    vecs = []
    for i in range(0, len(texts), batch):
        if i > 0:
            time.sleep(7)  # 개발키 RPM 10 제한 대비
        res = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch])
        vecs.extend(d.embedding for d in res.data)
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_index(recipes, cache="embeddings.npy"):
    """레시피 임베딩을 한 번만 만들고 파일로 캐시 (호출 수 절약)."""
    texts = [r["text"] for r in recipes]
    key = hashlib.md5((EMBED_MODEL + json.dumps(texts, ensure_ascii=False)).encode()).hexdigest()
    meta = cache + ".key"
    if os.path.exists(cache) and os.path.exists(meta) and open(meta).read() == key:
        return np.load(cache)
    vecs = _embed(texts)
    np.save(cache, vecs)
    open(meta, "w").write(key)
    return vecs


def retrieve(question, recipes, index=None, k=10):
    kw = keyword_scores(question, recipes)
    if index is not None:
        qv = _embed([question])[0]
        score = index @ qv + 0.1 * kw  # 의미 유사도 + 키워드 보너스
    else:
        score = kw
    if score.max() <= 0:
        return []
    top = np.argsort(-score)[:k]
    return [recipes[i] for i in top if score[i] > 0]


# ---------- 답변 ----------
def _clean(text):
    # 추론형 모델이 <think>...</think>를 붙여 보내는 경우 제거
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def answer(question, history, recipes, mode="full", index=None):
    """mode: 'full' = 전체 레시피 넣기 / 'rag' = 관련 레시피만 골라 넣기"""
    if mode == "rag":
        used = retrieve(question, recipes, index)
        if not used:  # 아무것도 못 찾으면 전체로 대체
            used = recipes
    else:
        used = recipes

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n[레시피 목록]\n" + format_context(used)}]
    messages += history[-MAX_HISTORY:]
    messages.append({"role": "user", "content": question})

    res = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.3)
    return _clean(res.choices[0].message.content), used
