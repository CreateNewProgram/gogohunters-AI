import os, requests
from dotenv import load_dotenv

load_dotenv()
# === 환경설정 ===
TYPECAST_API_KEY = os.getenv("TYPECAST_API_KEY")
url = "https://typecast.ai/api/actor"
headers = {
    "Authorization": f"Bearer {TYPECAST_API_KEY}",
    "Content-Type": "application/json"
}

# === 요청 보내기 ===
response = requests.get(url, headers=headers)

# === 응답 처리 ===
try:
    data = response.json()
except Exception as e:
    print(f"❌ JSON 파싱 실패: {e}")
    print(response.text)
    exit()

# === actor 정보 추출 ===
if "result" in data and isinstance(data["result"], list):
    print("✅ actor 목록:")
    for actor in data["result"]:
        name_ko = actor.get("name", {}).get("ko", "이름없음")
        actor_id = actor.get("actor_id", "없음")
        lang = actor.get("language", "알 수 없음")
        print(f"- 이름: {name_ko} | actor_id: {actor_id} | 언어: {lang}")
else:
    print("❗ 'result' 키가 없거나 리스트가 아닙니다.")
