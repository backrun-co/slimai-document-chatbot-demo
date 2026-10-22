"""Giao diện Streamlit cho demo chatbot hỏi đáp tài liệu nội bộ."""

from __future__ import annotations

import html
import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag import Chunk, NOT_FOUND_ANSWER, answer_question, create_embeddings, extract_document, retrieve


load_dotenv()

MAX_FILES = 5
MAX_FILE_SIZE = 10 * 1024 * 1024
SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md", "html", "htm"]
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

st.set_page_config(
    page_title="Trợ lý tài liệu nội bộ",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
    <style>
      :root {
        --acid: #dfff36;
        --acid-soft: #b9d91e;
        --ink: #070806;
        --panel: #0e100c;
        --text: #f5f7ee;
        --muted: #92998a;
      }
      html, body, [class*="css"] { color-scheme: dark; }
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at 78% 6%, rgba(205, 231, 0, .16), transparent 24rem),
          linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px),
          #070806;
        background-size: auto, 38px 38px, 38px 38px, auto;
        color: var(--text);
      }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
      [data-testid="stSidebar"] {
        background: rgba(9, 10, 8, .97);
        border-right: 1px solid rgba(255,255,255,.07);
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
      .block-container { max-width: 960px; padding-top: 1.2rem; padding-bottom: 7rem; }

      .side-brand, .top-brand { display: flex; align-items: center; gap: .65rem; }
      .side-brand { margin: 0 0 2rem; }
      .brand-mark {
        display: inline-flex; align-items: center; justify-content: center;
        width: 30px; height: 30px; border-radius: 9px;
        color: #080900; background: var(--acid); font-weight: 900;
        box-shadow: 0 0 22px rgba(223,255,54,.28);
      }
      .brand-name { color: var(--text); font-size: 1rem; font-weight: 760; letter-spacing: -.02em; }
      .brand-name small { color: var(--muted); font-size: .72rem; font-weight: 500; margin-left: .35rem; }
      .sidebar-label, .eyebrow {
        color: var(--muted); font-size: .71rem; font-weight: 700;
        letter-spacing: .13em; text-transform: uppercase;
      }
      .sidebar-label { margin: 0 0 .6rem; }
      .privacy-note { color: #70776b; font-size: .75rem; line-height: 1.5; margin: .8rem 0; }

      .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: .5rem; }
      .model-badge {
        display: inline-flex; align-items: center; gap: .45rem;
        color: #a4aa9c; font-size: .74rem; padding: .42rem .7rem;
        border: 1px solid rgba(255,255,255,.08); border-radius: 999px;
        background: rgba(14,16,12,.72);
      }
      .model-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--acid); box-shadow: 0 0 10px var(--acid); }

      .hero-shell { text-align: center; padding: 2.1rem 0 1.35rem; transition: padding .2s ease; }
      .hero-shell.compact { padding: 1rem 0 .5rem; }
      .knowledge-orb {
        width: 88px; height: 88px; margin: 0 auto 1.4rem; border-radius: 50%; position: relative;
        background:
          radial-gradient(circle at 35% 30%, rgba(255,255,255,.85) 0 2%, transparent 9%),
          radial-gradient(circle at 45% 45%, #e9ff47 0 7%, #8faa00 25%, #283000 54%, #080900 72%);
        border: 1px solid rgba(223,255,54,.75);
        box-shadow: 0 0 12px rgba(223,255,54,.45), 0 0 55px rgba(190,221,0,.18), inset 0 0 18px rgba(255,255,255,.18);
      }
      .knowledge-orb::before, .knowledge-orb::after {
        content: ""; position: absolute; inset: -8px; border-radius: 50%;
        border: 1px solid rgba(223,255,54,.22); transform: rotate(18deg) scaleY(.72);
      }
      .knowledge-orb::after { inset: -15px; opacity: .45; transform: rotate(-30deg) scaleY(.82); }
      .hero-shell.compact .knowledge-orb { width: 54px; height: 54px; margin-bottom: .9rem; }
      .eyebrow { color: var(--acid-soft); margin-bottom: .7rem; }
      .hero-title {
        color: var(--text); font-size: clamp(2.3rem, 5vw, 4.4rem); line-height: .98;
        font-weight: 800; margin: 0 auto .9rem; letter-spacing: -.055em; max-width: 820px;
      }
      .hero-title span { color: var(--acid); }
      .hero-shell.compact .hero-title { font-size: clamp(1.9rem, 3.5vw, 2.8rem); }
      .hero-copy { color: var(--muted); font-size: .98rem; max-width: 560px; line-height: 1.65; margin: 0 auto 1rem; }
      .status-pill {
        display: inline-flex; align-items: center; gap: .55rem;
        background: rgba(17,19,14,.88); border: 1px solid rgba(255,255,255,.08);
        border-radius: 999px; padding: .48rem .78rem; color: #b7bdaf; font-size: .78rem;
      }
      .status-pill.ready { border-color: rgba(223,255,54,.22); color: #ddeaaf; }
      .status-icon { width: 7px; height: 7px; border-radius: 50%; background: #676d61; }
      .ready .status-icon { background: var(--acid); box-shadow: 0 0 10px rgba(223,255,54,.8); }
      .empty-hint { color: #6f756a; font-size: .82rem; text-align: center; margin: .55rem 0 1.2rem; }

      [data-testid="stFileUploaderDropzone"] {
        background: #0d0f0b; border: 1px dashed rgba(223,255,54,.26); border-radius: 16px;
        padding: .45rem;
      }
      [data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(223,255,54,.55); }
      [data-testid="stFileUploaderDropzone"] button {
        background: #181b14 !important; border: 1px solid rgba(255,255,255,.09) !important;
        color: #e9ede1 !important; border-radius: 10px !important;
      }
      [data-testid="stFileUploaderDropzoneInstructions"] > div > span { color: #b7bdaf; }
      [data-testid="stFileUploaderDropzoneInstructions"] small { color: #666d61; }

      .stButton > button[kind="primary"] {
        background: var(--acid); color: #0b0c06; border: 0; border-radius: 12px;
        font-weight: 750; min-height: 2.8rem; box-shadow: 0 0 22px rgba(223,255,54,.12);
      }
      .stButton > button[kind="primary"]:hover { background: #ebff72; color: #070806; }
      .stButton > button:disabled,
      .stButton > button[kind="primary"]:disabled {
        background: #151711 !important; color: #596052 !important;
        border: 1px solid rgba(255,255,255,.055) !important; box-shadow: none !important;
      }
      .stButton > button:not([kind="primary"]) {
        background: #11130f; color: #aeb5a6; border: 1px solid rgba(255,255,255,.08); border-radius: 12px;
      }
      .doc-card {
        margin: .55rem 0; padding: .72rem .8rem; border-radius: 12px;
        background: #11130f; border: 1px solid rgba(255,255,255,.06);
      }
      .doc-name { color: #e8ebdf; font-size: .82rem; font-weight: 650; overflow-wrap: anywhere; }
      .doc-meta { color: #737a6e; font-size: .7rem; margin-top: .2rem; }

      [data-testid="stChatMessage"] {
        background: rgba(14,16,12,.9); border: 1px solid rgba(255,255,255,.07);
        border-radius: 18px; padding: .55rem .75rem; margin: .7rem 0;
        box-shadow: 0 14px 35px rgba(0,0,0,.2);
      }
      [data-testid="stChatMessageAvatarUser"] { background: #25281f; }
      [data-testid="stChatMessageAvatarAssistant"] { background: rgba(223,255,54,.13); }
      [data-testid="stChatInput"] {
        background: #090a08; border: 1px solid rgba(223,255,54,.18); border-radius: 20px;
        box-shadow: 0 18px 55px rgba(0,0,0,.55), 0 0 35px rgba(223,255,54,.045);
      }
      [data-testid="stChatInput"]:focus-within { border-color: rgba(223,255,54,.46); }
      [data-testid="stChatInput"] textarea { color: #edf0e7; }
      [data-testid="stChatInputSubmitButton"] { color: var(--acid); }
      [data-testid="stBottomBlockContainer"] { background: linear-gradient(transparent, #070806 38%); padding-top: 2rem; }
      [data-testid="stExpander"] { background: #0b0d09; border: 1px solid rgba(255,255,255,.07); border-radius: 13px; }
      .source-meta { color: var(--acid-soft); font-weight: 700; font-size: .78rem; }
      .source-text { color: #a9afa2; line-height: 1.62; font-size: .88rem; margin-top: .35rem; }
      [data-testid="stAlert"] { border-radius: 14px; background: #11130f; color: #c3c9ba; }
      hr { border-color: rgba(255,255,255,.07) !important; }
      footer { visibility: hidden; }

      @media (max-width: 760px) {
        .block-container { padding-top: .8rem; }
        .topbar { margin-bottom: 0; }
        .model-badge { display: none; }
        .hero-shell { padding-top: 1.2rem; }
        .knowledge-orb { width: 68px; height: 68px; }
        .hero-title { font-size: 2.6rem; }
        .hero-copy { font-size: .9rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "chunks": [],
        "embeddings": np.empty((0, 0), dtype=np.float32),
        "messages": [],
        "documents": [],
        "index_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_knowledge() -> None:
    st.session_state.chunks = []
    st.session_state.embeddings = np.empty((0, 0), dtype=np.float32)
    st.session_state.messages = []
    st.session_state.documents = []
    st.session_state.index_signature = None


def get_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return OpenAI(api_key=api_key) if api_key else None


def upload_signature(files: list) -> tuple | None:
    if not files:
        return None
    return tuple((item.name, item.size, hash(item.getvalue())) for item in files)


def show_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("Không có đoạn tài liệu đủ liên quan để làm nguồn.")
        return
    with st.expander(f"Xem {len(sources)} đoạn nguồn đã dùng"):
        for index, source in enumerate(sources, start=1):
            safe_filename = html.escape(source["filename"])
            safe_text = html.escape(source["text"])
            safe_location = html.escape(source.get("location") or f'Trang {source.get("page", "?")}')
            st.markdown(
                f'<div class="source-meta">Nguồn {index} · {safe_filename} · {safe_location}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="source-text">{safe_text}</div>', unsafe_allow_html=True)
            if index < len(sources):
                st.divider()


init_state()
client = get_client()

with st.sidebar:
    st.markdown(
        '<div class="side-brand"><span class="brand-mark">◈</span><span class="brand-name">DocQuery</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-label">Kho tài liệu</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Tải tài liệu",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not client:
        st.warning(
            "Chưa có OPENAI_API_KEY. Chạy local: thêm key vào tệp .env. "
            "Bản online: thêm key trong Manage app → Settings → Secrets."
        )
    else:
        st.markdown(
            '<div class="privacy-note">● Nội dung tệp được gửi tới OpenAI API để lập chỉ mục và trả lời.</div>',
            unsafe_allow_html=True,
        )

    signature = upload_signature(uploaded_files)
    can_index = bool(uploaded_files) and client is not None
    if st.button("Đọc và lập chỉ mục", type="primary", use_container_width=True, disabled=not can_index):
        if len(uploaded_files) > MAX_FILES:
            st.error(f"Chỉ được tải tối đa {MAX_FILES} tệp.")
        elif any(item.size > MAX_FILE_SIZE for item in uploaded_files):
            st.error("Có tệp vượt quá giới hạn 10 MB.")
        else:
            with st.spinner("Đang đọc, chia nhỏ và lập chỉ mục…"):
                try:
                    all_chunks: list[Chunk] = []
                    documents: list[dict] = []
                    for item in uploaded_files:
                        file_chunks, summary = extract_document(item.getvalue(), item.name)
                        all_chunks.extend(file_chunks)
                        documents.append(
                            {"name": item.name, "summary": summary, "chunks": len(file_chunks)}
                        )

                    if not all_chunks:
                        st.error("Không trích được văn bản. Tệp có thể rỗng, bị hỏng hoặc là PDF scan cần OCR.")
                    else:
                        embeddings = create_embeddings(
                            client,
                            [chunk.text for chunk in all_chunks],
                            EMBEDDING_MODEL,
                        )
                        st.session_state.chunks = all_chunks
                        st.session_state.embeddings = embeddings
                        st.session_state.documents = documents
                        st.session_state.messages = []
                        st.session_state.index_signature = signature
                        st.success(f"Đã sẵn sàng: {len(all_chunks)} đoạn từ {len(documents)} tệp.")
                except Exception as error:
                    st.error(f"Không thể xử lý tài liệu: {error}")

    if st.session_state.documents:
        st.markdown(
            '<div class="sidebar-label" style="margin-top:1.4rem">Tài liệu đang dùng</div>',
            unsafe_allow_html=True,
        )
        for document in st.session_state.documents:
            st.markdown(
                f'<div class="doc-card"><div class="doc-name">{html.escape(document["name"])}</div>'
                f'<div class="doc-meta">{html.escape(document["summary"])} · {document["chunks"]} đoạn chỉ mục</div></div>',
                unsafe_allow_html=True,
            )
        if st.button("Xóa phiên làm việc", use_container_width=True):
            clear_knowledge()
            st.rerun()

hero_class = "hero-shell compact" if st.session_state.messages else "hero-shell"
if st.session_state.chunks:
    status_html = (
        f'<div class="status-pill ready"><span class="status-icon"></span>'
        f'Đã sẵn sàng · {len(st.session_state.chunks)} đoạn được lập chỉ mục</div>'
    )
else:
    status_html = (
        '<div class="status-pill"><span class="status-icon"></span>Chưa có tài liệu</div>'
        '<div class="empty-hint">Tải PDF, Word, PowerPoint, Excel hoặc tệp văn bản để bắt đầu</div>'
    )

st.markdown(
    f'<section class="{hero_class}"><div class="knowledge-orb"></div>'
    '<div class="eyebrow">Trợ lý tài liệu nội bộ</div>'
    '<div class="hero-title">Hỏi tài liệu.<br><span>Nhận câu trả lời có nguồn.</span></div>'
    '<div class="hero-copy">Hỗ trợ PDF, Word, PowerPoint, Excel và tệp văn bản. Chỉ trả lời từ nội dung bạn cung cấp.</div>'
    f'{status_html}</section>',
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

question = st.chat_input(
    "Hỏi về chính sách, quy trình, số liệu trong tài liệu…",
    disabled=not bool(st.session_state.chunks) or client is None,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm trong tài liệu…"):
            try:
                results = retrieve(
                    client,
                    question,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    EMBEDDING_MODEL,
                    top_k=4,
                )

                relevant = results if results and results[0].score >= 0.30 else []
                answer = answer_question(
                    client,
                    question,
                    relevant,
                    CHAT_MODEL,
                    recent_messages=st.session_state.messages[:-1],
                )
                sources = []
                if answer.strip() != NOT_FOUND_ANSWER:
                    sources = [
                        {
                            "filename": item.chunk.filename,
                            "location": item.chunk.source_location,
                            "text": item.chunk.text,
                            "score": item.score,
                        }
                        for item in relevant
                    ]
                st.markdown(answer)
                show_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as error:
                st.error(f"Không thể trả lời lúc này: {error}")
