import asyncio
import websockets
import json
import os

async def send_audio_file(websocket, file_path):
    if not os.path.exists(file_path):
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return

    # 1. 오디오 전송
    with open(file_path, "rb") as f:
        audio_data = f.read()
    await websocket.send(audio_data)
    print("✅ 오디오 전송 완료")

    # 2. JSON 응답 수신
    response = await websocket.recv()
    if isinstance(response, str):
        try:
            response_json = json.loads(response)
            print("📝 JSON 응답:", response_json)
        except json.JSONDecodeError:
            print("❌ JSON 디코딩 실패:", response)
            return
    else:
        print("❌ 예상과 다른 JSON 응답:", type(response))
        return

    # 3. 오디오 응답 수신
    audio_bytes = await websocket.recv()
    if not isinstance(audio_bytes, bytes):
        print("❌ 오디오 데이터가 bytes 형식이 아님")
        return

    # 4. 저장 및 재생
    output_path = "received_from_server.wav"
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    print(f"🔊 오디오 저장 완료: {output_path}")

async def run_client():
    uri = "wss://manga-textbooks-original-distribution.trycloudflare.com/ws"

    try:
        async with websockets.connect(
            uri,
            ping_interval=None,
            max_size=10 * 1024 * 1024
        ) as websocket:
            print("🔗 WebSocket 연결 완료")

            while True:
                file_path = input("\n🎙 전송할 WAV 파일 경로 (종료하려면 'q'): ").strip()
                if file_path.lower() == "q":
                    print("👋 종료")
                    break

                await send_audio_file(websocket, file_path)

    except Exception as e:
        print(f"❌ 연결 또는 전송 중 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(run_client())
