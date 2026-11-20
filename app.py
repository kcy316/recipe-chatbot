"""Streamlit 채팅 화면. 실행: streamlit run app.py"""
import streamlit as st

import rag

st.set_page_config(page_title="집밥 레시피 Q&A", page_icon="🍳")
st.title("🍳 집밥 레시피 Q&A")


@st.cache_data
def get_recipes():
    return rag.load_recipes()


recipes = get_recipes()

with st.sidebar:
    st.caption(f"모델: `{rag.CHAT_MODEL}` · 레시피 {len(recipes)}개")
    options = ["전체 목록 넣기", "검색(RAG)"]
    mode_label = st.radio("검색 방식", options)
    mode = "full" if mode_label == options[0] else "rag"
    if mode == "rag":
        st.caption("임베딩: " + (f"`{rag.EMBED_MODEL}`" if rag.EMBED_MODEL else "미설정 → 키워드 검색"))
    st.metric("이번 세션 API 호출", st.session_state.get("calls", 0))
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

index = None
if mode == "rag" and rag.EMBED_MODEL:
    if "index" not in st.session_state:
        with st.spinner("레시피 임베딩 만드는 중(최초 1회)…"):
            st.session_state.index = rag.build_index(recipes)
    index = st.session_state.index

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("예: 감자랑 양파로 만들 수 있는 메뉴 알려줘"):
    st.chat_message("user").markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("생각 중…"):
            try:
                text, used = rag.answer(q, st.session_state.messages, recipes, mode, index)
                st.session_state.calls = st.session_state.get("calls", 0) + 1
            except Exception as e:
                text, used = f"⚠️ 호출 실패: {e}", []
        st.markdown(text)
        if mode == "rag" and used:
            with st.expander(f"참고한 레시피 {len(used)}개"):
                st.write("\n".join(f"- {r['text']}" for r in used))
    st.session_state.messages += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]
