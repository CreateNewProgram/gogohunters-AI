import os
import io
import time
import requests
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydub import AudioSegment
import numpy as np
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

# 사용자별 Gemini 세션 저장소
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

    try:
        json_meta = await websocket.receive_json()
        user_id = json_meta.get("user_id", "unknown")
        timestamp = json_meta.get("timestamp", "")
        print("✅ 메타데이터 수신:", json_meta)

        audio_binary = await websocket.receive_bytes()
        print(f"✅ 오디오 수신 완료 ({len(audio_binary)} bytes)")

        # 오디오 전처리
        audio = AudioSegment.from_file(io.BytesIO(audio_binary))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio_path = "temp_audio.wav"
        audio.export(audio_path, format="wav")

        print("🟡 Faster-Whisper STT 변환 중...")
        segments, _ = whisper_model.transcribe(audio_path, language="ko", beam_size=1)
        transcribed_text = "".join([seg.text for seg in segments]).strip()
        print(f"📝 STT 결과: {transcribed_text}")

        # Gemini 챗봇 세션 처리
        if user_id not in user_sessions:
            user_sessions[user_id] = gemini.start_chat(history=[])
            print(f"🟢 새로운 Gemini 세션 생성: {user_id}")
        else:
            print(f"🟢 기존 Gemini 세션 사용: {user_id}")
        chat = user_sessions[user_id]
        sequence = len(chat.history) // 2

        print("🟡 Gemini 응답 생성 중...")
        prompt = f"'{transcribed_text}' 라는 발화에 대해 아이들 대상으로 이해 할 수 있게 100자 내로 간결하게 한국어로 답변해줘."
        gemini_response = chat.send_message(prompt)
        answer_text = gemini_response.text.strip()
        print(f"📝 Gemini 응답: {answer_text}")

        print("🟡 Typecast TTS 요청 중...")
        print(f"🟡{TYPECAST_API_KEY},{TYPECAST_ACTOR_ID}")
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
        print(f"🟢 TTS 요청 완료: {speak_v2_url}")

        max_retry = 20
        for i in range(max_retry):
            check = requests.get(speak_v2_url, headers=tts_headers)
            check.raise_for_status()
            result = check.json()["result"]
            status = result["status"]

            if status == "done":
                audio_url = result["audio_download_url"]
                print("🟢 오디오 준비 완료:", audio_url)
                break
            elif status == "failed":
                raise Exception("❌ TTS 실패")
            else:
                print(f"⏳ TTS 대기 중... ({i+1}/{max_retry}) status={status}")
                time.sleep(1)
        else:
            raise TimeoutError("❌ TTS 응답이 제한 시간 내 도착하지 않았습니다.")

        audio_data = requests.get(audio_url)
        print(f"✅ 오디오 다운로드 완료 ({len(audio_data.content)} bytes)")

        with open("typecast_api_result.wav", "wb") as f:
            f.write(audio_data.content)
        print("📝 디버그용 오디오 저장 완료: typecast_api_result.wav")

        response_meta = {
            "user_id": user_id,
            "sequence": sequence,
            "timestamp": timestamp,
            "answer": answer_text
        }
        await websocket.send_json(response_meta)
        await websocket.send_bytes(audio_data.content)
        print("✅ WebSocket 응답 전송 완료")

    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        await websocket.send_text("❌ 서버 오류: " + str(e))

    finally:
        await websocket.close()
        print("🔴 클라이언트 연결 종료")

# ─────────────── 세션 목록 보기 ───────────────
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

    history = []
    try:
        for item in session.history:
            role = getattr(item, "role", "unknown")

            if hasattr(item, "parts"):
                if isinstance(item.parts, list):
                    content = "\n".join(str(p) for p in item.parts)
                else:
                    content = str(item.parts)
            else:
                content = "내용 없음"

            history.append({"role": role, "content": content})
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": "히스토리 파싱 중 오류 발생",
            "details": str(e)
        })

    return JSONResponse(content={
        "user_id": user_id,
        "message_count": len(history),
        "chat_history": history
    })


# uvicorn main:app --reload --port 8090
#  cloudflared tunnel --url http://localhost:8090