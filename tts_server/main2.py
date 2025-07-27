import os
import io
import time
import uuid
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydub import AudioSegment
import torch
from faster_whisper import WhisperModel
from google.generativeai import GenerativeModel, configure

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
                prompt = f"'{transcribed_text}' 라는 발화에 대해 아이들 대상으로 이해할 수 있게 100자 내로 간결하게 한국어로 답변해줘."
                gemini_response = chat.send_message(prompt)
                answer_text = gemini_response.text.strip()
                print(f"🤖 Gemini 응답: {answer_text}")

                # 🗣️ Typecast TTS 요청
                tts_headers = {
                    "Authorization": f"Bearer {TYPECAST_API_KEY}",
                    "Content-Type": "application/json"
                }
                tts_payload = {
                    "text": answer_text,
                    "lang": "auto",
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
                tts_response.raise_for_status()
                speak_v2_url = tts_response.json()["result"]["speak_v2_url"]

                for i in range(20):
                    check = requests.get(speak_v2_url, headers=tts_headers)
                    status = check.json()["result"]["status"]
                    if status == "done":
                        audio_url = check.json()["result"]["audio_download_url"]
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
