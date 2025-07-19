import os
import io
import time
import requests
import numpy as np
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from google.generativeai import GenerativeModel, configure

# ─────────────── 환경 변수 로드 ───────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TYPECAST_API_KEY = os.getenv("TYPECAST_API_KEY")
TYPECAST_ACTOR_ID = os.getenv("TYPECAST_ACTOR_ID")

# ─────────────── Gemini 초기화 ───────────────
configure(api_key=GEMINI_API_KEY)
gemini = GenerativeModel("models/gemini-1.5-pro")

# 사용자별 Gemini 세션 저장소
user_sessions = {}

# ─────────────── FastAPI 앱 ───────────────
app = FastAPI()

# ─────────────── Whisper 초기화 ───────────────
print("🟡 Whisper 초기화 시작")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-small"

print(f"🟡 모델 로드 중: {model_id} (device={device})")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)
print("🟢 Whisper 로드 완료")

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

        audio = AudioSegment.from_file(io.BytesIO(audio_binary))
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        print("🟡 Whisper STT 변환 중...")
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt").to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
        with torch.no_grad():
            generated_ids = model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)
        transcribed_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"📝 STT 결과: {transcribed_text}")

        if user_id not in user_sessions:
            user_sessions[user_id] = gemini.start_chat(history=[])
            print(f"🟢 새로운 Gemini 세션 생성: {user_id}")
        else:
            print(f"🟢 기존 Gemini 세션 사용: {user_id}")
        chat = user_sessions[user_id]

        # ✅ 시퀀스는 현재 대화 기록 수로 자동 계산
        sequence = len(chat.history) // 2

        print("🟡 Gemini 응답 생성 중...")
        prompt = f"'{transcribed_text}' 라는 발화에 대해 핵심만 간결하게 한국어로 답변해줘."
        gemini_response = chat.send_message(prompt)
        answer_text = gemini_response.text.strip()
        print(f"📝 Gemini 응답: {answer_text}")

        print("🟡 Typecast TTS 요청 중...")
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
            "xapi_hd": True
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

            # parts: Content 객체나 str이 섞여 있을 수 있음
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
