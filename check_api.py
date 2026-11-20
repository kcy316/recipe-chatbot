"""API 연결 확인용. 가장 먼저 실행: python check_api.py"""
import rag

print("BASE_URL :", rag.BASE_URL)
print("API_KEY  :", (rag.API_KEY[:7] + "…") if rag.API_KEY else "(없음! .env 확인)")

print("\n[1] 사용 가능한 모델 목록")
try:
    for m in rag.client.models.list().data:
        print("  -", m.id)
except Exception as e:
    print("  모델 목록 조회 실패(서버가 지원 안 할 수도 있음):", e)

print(f"\n[2] 채팅 테스트 ({rag.CHAT_MODEL})")
try:
    r = rag.client.chat.completions.create(
        model=rag.CHAT_MODEL, messages=[{"role": "user", "content": "한 문장으로 자기소개 해줘"}])
    print("  ✅", r.choices[0].message.content)
except Exception as e:
    print("  ❌", e)

if rag.EMBED_MODEL:
    print(f"\n[3] 임베딩 테스트 ({rag.EMBED_MODEL})")
    try:
        r = rag.client.embeddings.create(model=rag.EMBED_MODEL, input=["감자조림"])
        print("  ✅ 벡터 차원:", len(r.data[0].embedding))
    except Exception as e:
        print("  ❌", e)
