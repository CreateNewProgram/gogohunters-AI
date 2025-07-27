from fastapi import FastAPI, WebSocket
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from gtts import gTTS
from pydub import AudioSegment
import numpy as np
import io
import os
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure

# ─────────────────────── 환경변수 로드 ───────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ─────────────────────── Gemini 초기화 ───────────────────────
configure(api_key=GEMINI_API_KEY)
gemini = GenerativeModel("models/gemini-1.5-pro")

# ─────────────────────── FastAPI ───────────────────────
app = FastAPI()

# ─────────────────────── Whisper STT 초기화 ───────────────────────
print("🟡 Whisper 초기화 시작")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-small"

print(f"🟡 모델 로드 중: {model_id} (device={device})")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
print("🟢 모델 로드 완료")

print("🟡 프로세서 로드 중...")
processor = AutoProcessor.from_pretrained(model_id)
print("🟢 프로세서 로드 완료")

# ─────────────────────── WebSocket 엔드포인트 ───────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ 클라이언트 연결됨")

    try:
        # 1회만 처리
        print("🟡 JSON 메타데이터 수신 대기 중...")
        json_meta = await websocket.receive_json()
        print("✅ 메타데이터 수신:", json_meta)

        print("🟡 음성 바이너리 수신 대기 중...")
        audio_binary = await websocket.receive_bytes()
        print(f"✅ 오디오 {len(audio_binary)} bytes 수신 완료")

        # pydub 로 wav/pcm 디코딩
        audio = AudioSegment.from_file(io.BytesIO(audio_binary))
        audio = audio.set_frame_rate(16000).set_channels(1)  # Whisper 권장 16kHz mono
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        # STT
        print("🟡 STT 변환 시작...")
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt").to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")

        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                forced_decoder_ids=forced_decoder_ids
            )
        transcribed_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"📝 STT 결과: {transcribed_text}")

        # 챗봇(Gemini)
        print("🟡 Gemini 응답 생성 중...")
        chat = gemini.start_chat(history=[])
        prompt = f"'{transcribed_text}' 라는 발화에 대해 핵심만 간결하게 한국어로 답변해줘."
        gemini_response = chat.send_message(prompt)
        answer_text = gemini_response.text
        print(f"📝 Gemini 응답: {answer_text}")

        # TTS
        print("🟡 TTS 변환(gTTS) 시작...")
        tts = gTTS(text=answer_text, lang="ko")
        tts.save("output.mp3")
        print("🟢 TTS 변환 완료, mp3 생성됨")

        print("🟡 mp3 → PCM 변환 시작...")
        sound = AudioSegment.from_mp3("output.mp3")
        pcm_io = io.BytesIO()
        sound.export(pcm_io, format="wav", parameters=["-acodec", "pcm_s16le"])
        pcm_io.seek(0)
        print("🟢 PCM 변환 완료")

        response_meta = {
            "user_id": json_meta["user_id"],
            "sequence": json_meta["sequence"],
            "timestamp": json_meta["timestamp"],
            "answer": answer_text
        }
        print("🟡 응답 JSON 전송 중...")
        await websocket.send_json(response_meta)
        print("🟡 음성 데이터(PCM) 전송 중...")
        await websocket.send_bytes(pcm_io.read())

        print("✅ 응답 전송 완료")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    finally:
        print("🔴 클라이언트 연결 종료")
        await websocket.close()

# ─────────────────────── 실행 ───────────────────────
if __name__ == "__main__":
    import uvicorn
    print("🚀 서버 실행 준비완료")
    uvicorn.run("chat_server_gemini:app", host="127.0.0.1", port=8090, reload=True)

#  cloudflared tunnel --url http://localhost:8090