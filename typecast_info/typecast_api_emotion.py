import os ,requests
import json
from dotenv import load_dotenv

# 실제 Typecast API 키와 actor_id를 여기에 입력
load_dotenv()
TYPECAST_API_KEY = os.getenv("TYPECAST_API_KEY")
TYPECAST_ACTOR_ID = os.getenv("TYPECAST_ACTOR_ID")

url = f"https://typecast.ai/api/actor/{TYPECAST_ACTOR_ID}/versions"
headers = {
    "Authorization": f"Bearer {TYPECAST_API_KEY}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("✅ 사용 가능한 감정 tone presets:")
    for version in data["result"]:
        if "latest" in version.get("aliases", []):
            print("▶ latest version:", version["display_name"])
            print("  emotions:", version.get("emotion_tone_presets", []))
else:
    print(f"❌ 에러 발생 ({response.status_code}): {response.text}")
