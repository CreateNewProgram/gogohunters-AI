import os
import io
import time
import uuid
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
# ─────────────── RAG 환경 불러오기 ───────────────
# 🔁 사전 준비: PDF 문서 불러오기
def extract_text_from_pdfs(pdf_dir_path="pdfs"):
    pdf_dir = Path(pdf_dir_path)
    docs = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        doc = fitz.open(pdf_file)
        text = ""
        for page in doc:
            text += page.get_text()
        docs.append({
            "filename": pdf_file.name,
            "content": text
        })
        doc.close()
    return docs

# 🔁 사전 준비: FAISS 인덱스 생성
def build_faiss_index(docs, model):
    texts = [doc["content"][:2048] for doc in docs]
    embeddings = model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# 🔍 문서 검색
def search_similar_docs(query, index, docs, model, top_k=3):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    results = [docs[i]["content"][:1000] for i in I[0]]
    return "\n\n".join(results)

# 🧠 Gemini RAG 응답
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

# 📦 서버 구동 시 메모리에 로드
print("🟡 RAG 초기화 시작")
docs = extract_text_from_pdfs("pdfs")
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
index, embeddings = build_faiss_index(docs, model)
print("🟢 RAG 로드 완료")
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

                # 💬 Gemini 응답 생성
                context_text = search_similar_docs(transcribed_text, index, docs, model)
                answer_text = ask_with_context(transcribed_text, context_text)
                print(f"🤖 Gemini 응답: {answer_text}")

                # 🗣️ Typecast TTS 요청
                tts_headers = {
                    "Authorization": f"Bearer {TYPECAST_API_KEY}",
                    "Content-Type": "application/json"
                }
                tts_payload = {
                    "text": answer_text[:300],
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

                # 예외 처리 추가
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
