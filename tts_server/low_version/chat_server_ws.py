from fastapi import FastAPI, WebSocket
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from gtts import gTTS
from pydub import AudioSegment
import io

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

print("🟡 파이프라인 초기화 중...")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
print("🟢 파이프라인 초기화 완료")

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

        # STT
        print("🟡 STT 변환 시작...")
        result = pipe(audio_binary)
        transcribed_text = result["text"]
        print(f"📝 STT 결과: {transcribed_text}")

        # 챗봇
        print("🟡 챗봇 응답 생성 중...")
        answer_text = f"너가 말한 '{transcribed_text}' 의 의미를 설명해줄게."
        print(f"📝 챗봇 응답: {answer_text}")

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
    uvicorn.run("chat_server_ws:app", host="127.0.0.1", port=8090, reload=True)
