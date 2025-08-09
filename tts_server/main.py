import os
import io
import time
import uuid
import json
import hashlib
import requests
import faiss
import fitz
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydub import AudioSegment
from faster_whisper import WhisperModel
from google.generativeai import GenerativeModel, configure
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ─────────────── 환경 변수 로드 ───────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TYPECAST_API_KEY = os.getenv("TYPECAST_API_KEY")
TYPECAST_ACTOR_ID = os.getenv("TYPECAST_ACTOR_ID")

# ─────────────── Gemini 초기화 ───────────────
configure(api_key=GEMINI_API_KEY)
gemini = GenerativeModel("models/gemini-2.5-flash-lite")
user_sessions = {}

# ─────────────── FastAPI 앱 ───────────────
app = FastAPI()

# ─────────────── Faster-Whisper 초기화 ───────────────
print("🟡 Faster-Whisper 초기화 시작")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel(
    model_size_or_path="small",
    device=device,
    compute_type="int8" if device == "cuda" else "float32"
)
print("🟢 Faster-Whisper 로드 완료")

# ─────────────── RAG 설정 (경로/파라미터) ───────────────
PDF_DIR = Path("pdfs")
STORE_DIR = Path(".rag_store")
STORE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = STORE_DIR / "faiss.index"
META_PATH = STORE_DIR / "meta.json"

EMBED_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 5
MAX_CTX_CHARS = 2000

# ─────────────── 유틸: PDF 텍스트 추출 ───────────────
def extract_text_from_pdfs(pdf_dir_path="pdfs"):
    pdf_dir = Path(pdf_dir_path)
    docs = []
    if not pdf_dir.exists():
        print(f"⚠️ PDF 디렉토리가 없습니다: {pdf_dir.resolve()}")
        return docs

    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            doc = fitz.open(pdf_file)
            text = ""
            for page in doc:
                text += page.get_text()
            docs.append({"filename": pdf_file.name, "content": text})
            doc.close()
        except Exception as e:
            print(f"⚠️ PDF 파싱 실패: {pdf_file.name} - {e}")
    return docs

# ─────────────── 유틸: 텍스트 청킹 ───────────────
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    n = len(text)
    if n == 0:
        return chunks
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

# ─────────────── 유틸: 코퍼스 지문 ───────────────
def compute_corpus_fingerprint(pdf_dir: Path) -> str:
    """파일명, 크기, 수정시간으로 해시 생성 (내용 변경 감지)"""
    h = hashlib.sha256()
    if not pdf_dir.exists():
        return h.hexdigest()
    for p in sorted(pdf_dir.glob("*.pdf")):
        stat = p.stat()
        h.update(p.name.encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()

# ─────────────── 인덱스 저장/로드 ───────────────
def save_index(index, chunks_meta, fingerprint: str, params: dict):
    faiss.write_index(index, str(INDEX_PATH))
    meta = {
        "fingerprint": fingerprint,
        "chunks_meta": chunks_meta,
        "params": params,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print("💾 인덱스 저장 완료")

def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, None, None, None
    try:
        index = faiss.read_index(str(INDEX_PATH))
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        chunks_meta = meta.get("chunks_meta", [])
        fingerprint = meta.get("fingerprint")
        params = meta.get("params", {})
        return index, chunks_meta, fingerprint, params
    except Exception as e:
        print("⚠️ 인덱스 로드 실패:", e)
        return None, None, None, None

# ─────────────── FAISS 빌드 ───────────────
def build_faiss_index(docs, model, chunk_size=800, overlap=200):
    """
    반환:
      index: FAISS IndexFlatL2
      chunks_meta: [{doc_id, filename, text}]
      embeddings: np.ndarray (num_chunks, dim)
    """
    chunks_meta = []
    for i, doc in enumerate(docs):
        for ch in chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap):
            if ch.strip():
                chunks_meta.append({"doc_id": i, "filename": doc["filename"], "text": ch})

    if not chunks_meta:
        # 빈 인덱스 방지: 차원 알아내려면 dummy 임베딩
        dummy_vec = model.encode([" "], convert_to_numpy=True)
        dim = dummy_vec.shape[1]
        index = faiss.IndexFlatL2(dim)
        return index, [], np.zeros((0, dim), dtype=np.float32)

    texts = [c["text"] for c in chunks_meta]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index, chunks_meta, embeddings

# ─────────────── 검색 ───────────────
def search_similar_docs(query, index, chunks_meta, model, top_k=5, max_total_chars=2000):
    if not chunks_meta:
        return ""
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)

    picked, total_len = [], 0
    for idx in I[0]:
        c = chunks_meta[int(idx)]
        snippet = c["text"].strip()
        add_len = len(snippet)
        if total_len + add_len > max_total_chars:
            snippet = snippet[: max(0, max_total_chars - total_len)]
        picked.append(f"[{c['filename']}] {snippet}")
        total_len += len(snippet)
        if total_len >= max_total_chars:
            break
    return "\n\n".join(picked)

# ─────────────── Gemini RAG 응답 ───────────────
def ask_with_context(user_input, context_text):
    prompt = f"""
아래는 사용자의 질문과 관련된 참고 문서입니다. 이 문서를 바탕으로 질문에 답변해 주세요.
----- 참고 문서 -----
{context_text}
---------------------
질문: {user_input}
아이들도 이해할 수 있도록 100자 이내로 쉽고 정확하게 설명해 주세요.
"""
    response = gemini.generate_content(prompt)
    return response.text.strip()

# ─────────────── RAG 초기화 (캐시 사용) ───────────────
def init_rag(force_rebuild: bool = False):
    print("🟡 RAG 초기화 시작")
    current_fp = compute_corpus_fingerprint(PDF_DIR)

    # 캐시 로드 시도
    if not force_rebuild:
        cached = load_index()
    else:
        cached = (None, None, None, None)

    # 캐시 유효성 확인
    if cached and all(cached[:3]):
        cached_index, cached_chunks, cached_fp, cached_params = cached
        # 파라미터/모델/코퍼스 지문 일치 확인
        if (
            cached_fp == current_fp
            and cached_params
            and cached_params.get("embed_model_name") == EMBED_MODEL_NAME
            and cached_params.get("chunk_size") == CHUNK_SIZE
            and cached_params.get("chunk_overlap") == CHUNK_OVERLAP
        ):
            print(f"✅ 캐시 적중: 청크 {len(cached_chunks)}개, 인덱스 재사용")
            return cached_index, cached_chunks

    # 재빌드
    print("🔄 인덱스 재빌드 (문서 변경/파라미터 변경/캐시 없음)")
    docs = extract_text_from_pdfs(str(PDF_DIR))
    model = SentenceTransformer(EMBED_MODEL_NAME)
    index, chunks_meta, _ = build_faiss_index(
        docs, model, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
    )

    params = {
        "embed_model_name": EMBED_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "max_ctx_chars": MAX_CTX_CHARS,
    }
    save_index(index, chunks_meta, current_fp, params)
    print(f"🟢 RAG 로드 완료 (문서 {len(docs)}개, 청크 {len(chunks_meta)}개)")
    return index, chunks_meta

# ─────────────── 전역 RAG 핸들 ───────────────
# 주의: 검색 때 쓸 SentenceTransformer 모델은 한 번만 로드(메모리 상주)
rag_model = SentenceTransformer(EMBED_MODEL_NAME)
faiss_index, chunks_meta = init_rag(force_rebuild=False)

# ─────────────── WebSocket 엔드포인트 ───────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ 클라이언트 연결됨")

    user_id = str(uuid.uuid4())
    chat = gemini.start_chat(history=[])
    user_sessions[user_id] = chat
    print(f"🟢 Gemini 세션 생성됨: {user_id}")

    try:
        while True:
            try:
                # 🎧 오디오 수신
                audio_binary = await websocket.receive_bytes()
                print(f"🎧 오디오 수신 ({len(audio_binary)} bytes)")

                # 🔊 STT 전처리
                audio = AudioSegment.from_file(io.BytesIO(audio_binary))
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio_path = "temp_audio.wav"
                audio.export(audio_path, format="wav")

                segments, _ = whisper_model.transcribe(audio_path, language="ko", beam_size=1)
                transcribed_text = "".join([seg.text for seg in segments]).strip()
                print(f"📝 STT 결과: {transcribed_text}")

                # 💬 Gemini RAG: 검색 → 응답
                context_text = search_similar_docs(
                    transcribed_text, faiss_index, chunks_meta, rag_model, top_k=TOP_K, max_total_chars=MAX_CTX_CHARS
                )
                answer_text = ask_with_context(transcribed_text, context_text)
                print(f"🤖 Gemini 응답: {answer_text}")

                # 🗣️ Typecast TTS 요청
                tts_headers = {
                    "Authorization": f"Bearer {TYPECAST_API_KEY}",
                    "Content-Type": "application/json"
                }
                tts_payload = {
                    "text": answer_text[:500],
                    "lang": "ko-kr",
                    "tts_mode": "actor",
                    "actor_id": TYPECAST_ACTOR_ID,
                    "model_version": "latest",
                    "xapi_audio_format": "wav",
                    "xapi_hd": True,
                    "volume": 100,
                    "speed_x": 1,
                    "tempo": 1,
                    "pitch": 0
                }

                tts_response = requests.post("https://typecast.ai/api/speak", headers=tts_headers, json=tts_payload)

                # 예외 처리
                if tts_response.status_code == 400:
                    print(f"❌ [400 Bad Request] 응답: {tts_response.text}")
                    raise Exception("TTS 요청에 필요한 파라미터가 누락되었거나 잘못되었습니다.")
                elif tts_response.status_code == 401:
                    print(f"❌ [401 Unauthorized] 응답: {tts_response.text}")
                    raise Exception("TTS 인증 실패: API 키를 확인하세요.")
                elif tts_response.status_code == 429:
                    print(f"❌ [429 Too Many Requests] 응답: {tts_response.text}")
                    raise Exception("TTS 요청이 너무 많습니다. 잠시 후 다시 시도해주세요.")
                elif not tts_response.ok:
                    print(f"❌ [TTS 요청 실패] 상태 코드: {tts_response.status_code}, 응답: {tts_response.text}")
                    raise Exception(f"TTS 요청 실패: {tts_response.status_code}")

                speak_v2_url = tts_response.json()["result"]["speak_v2_url"]

                # 상태 폴링
                audio_url = None
                for _ in range(20):
                    check = requests.get(speak_v2_url, headers=tts_headers)
                    result = check.json()["result"]
                    status = result["status"]
                    if status == "done":
                        audio_url = result["audio_download_url"]
                        break
                    elif status == "failed":
                        raise Exception("❌ TTS 처리 실패")
                    time.sleep(1)
                else:
                    raise TimeoutError("❌ TTS 대기 시간 초과")

                audio_data = requests.get(audio_url).content

                # 📤 응답 전송
                await websocket.send_json({
                    "user_id": user_id,
                    "text": transcribed_text,
                    "answer": answer_text,
                    "timestamp": time.time()
                })
                await websocket.send_bytes(audio_data)
                print("✅ 응답 전송 완료")

            except WebSocketDisconnect:
                print("🔴 클라이언트 연결 해제됨 (WebSocketDisconnect)")
                break

            except Exception as e_inner:
                print(f"⚠️ 처리 중 오류 발생: {e_inner}")
                if websocket.client_state.name == "CONNECTED":
                    try:
                        await websocket.send_text("❌ 오류: " + str(e_inner))
                    except Exception as send_fail:
                        print(f"❌ 오류 메시지 전송 실패: {send_fail}")
                break

    finally:
        if websocket.client_state.name != "DISCONNECTED":
            try:
                await websocket.close()
                print("🔒 WebSocket 닫힘")
            except Exception as close_error:
                print(f"⚠️ WebSocket 종료 실패: {close_error}")

        print("🧹 세션 종료:", user_id)
        user_sessions.pop(user_id, None)

# ─────────────── 세션 목록 조회 ───────────────
@app.get("/sessions")
async def sessions(request: Request):
    user_id = request.query_params.get("user_id")
    if user_id is None:
        return JSONResponse(content={
            "active_sessions_count": len(user_sessions),
            "user_ids": list(user_sessions.keys())
        })

    session = user_sessions.get(user_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "세션을 찾을 수 없습니다."})

    try:
        history = []
        for item in session.history:
            role = getattr(item, "role", "unknown")
            content = (
                "\n".join(str(p) for p in item.parts)
                if hasattr(item, "parts") and isinstance(item.parts, list)
                else str(getattr(item, "parts", "내용 없음"))
            )
            history.append({"role": role, "content": content})
        return JSONResponse(content={
            "user_id": user_id,
            "message_count": len(history),
            "chat_history": history
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": "히스토리 파싱 중 오류 발생",
            "details": str(e)
        })

# 실행 명령 예시
# uvicorn main:app --reload --port 8090
# cloudflared tunnel --url http://localhost:8090
