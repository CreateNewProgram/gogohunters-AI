import os
import io
import time
import requests
import numpy as np
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from google.generativeai import GenerativeModel, configure

# ─────────────── 환경 변수 로드 ───────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
V1_TYPECAST_API_KEY = os.getenv("V1_TYPECAST_API_KEY")
V1_TYPECAST_VOICE_ID = os.getenv("V1_TYPECAST_VOICE_ID")

# ─────────────── Gemini 초기화 ───────────────
configure(api_key=GEMINI_API_KEY)
gemini = GenerativeModel("models/gemini-1.5-pro")

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
        # 1. 메타데이터 수신
        json_meta = await websocket.receive_json()
        print("✅ 메타데이터 수신:", json_meta)

        # 2. 오디오 수신
        audio_binary = await websocket.receive_bytes()
        print(f"✅ 오디오 {len(audio_binary)} bytes 수신 완료")

        # 3. 오디오 전처리
        audio = AudioSegment.from_file(io.BytesIO(audio_binary))
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        # 4. Whisper STT
        print("🟡 STT 변환 중...")
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt").to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
        with torch.no_grad():
            generated_ids = model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)
        transcribed_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"📝 STT 결과: {transcribed_text}")

        # 5. Gemini 응답 생성
        chat = gemini.start_chat(history=[])
        prompt = f"'{transcribed_text}' 라는 발화에 대해 핵심만 간결하게 한국어로 답변해줘."
        gemini_response = chat.send_message(prompt)
        answer_text = gemini_response.text.strip()
        print(f"📝 Gemini 응답: {answer_text}")

        # 6. Typecast TTS 요청 (v1 API)
        print("🟡 TTS 요청 중 (Typecast v1)...")
        tts_url = "https://api.typecast.ai/v1/text-to-speech"
        headers = {
            "X-API-KEY": V1_TYPECAST_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "voice_id": V1_TYPECAST_VOICE_ID,
            "text": answer_text,
            "model": "ssfm-v21",
            "language": "KOR",
            "prompt": {
                "emotion_preset": "normal",
                "emotion_intensity": 1.0
            },
            "output": {
                "volume": 100,
                "audio_pitch": 0,
                "audio_tempo": 1.0,
                "audio_format": "wav"
            },
            "seed": 42
        }

        tts_response = requests.post(tts_url, json=payload, headers=headers)
        if tts_response.status_code != 200:
            raise Exception(f"❌ TTS 요청 실패: {tts_response.text}")
        audio_data = tts_response.content
        print(f"🟢 TTS 응답 수신 완료 ({len(audio_data)} bytes)")

        # (선택) 디버그용 저장
        with open("typeast_v1_api_result.wav", "wb") as f:
            f.write(audio_data)
        print("📝 디버그용 오디오 저장 완료: typeast_v1_api_result.wav")

        # 7. WebSocket 응답 전송
        response_meta = {
            "user_id": json_meta.get("user_id", "unknown"),
            "sequence": json_meta.get("sequence", 1),
            "timestamp": json_meta.get("timestamp", ""),
            "answer": answer_text
        }
        await websocket.send_json(response_meta)
        await websocket.send_bytes(audio_data)
        print("✅ WebSocket 응답 전송 완료")

    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        await websocket.send_text("❌ 서버 오류: " + str(e))

    finally:
        await websocket.close()
        print("🔴 클라이언트 연결 종료")

# cloudflared tunnel --url http://localhost:8090

# uvicorn typecast_v1_api:app --port 8090