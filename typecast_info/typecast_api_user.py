import os, requests
from dotenv import load_dotenv

load_dotenv()
TYPECAST_API_KEY = os.getenv("TYPECAST_API_KEY")
url = "https://typecast.ai/api/me"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TYPECAST_API_KEY}"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("✅ 사용자 정보:")
    print(response.json())
else:
    print(f"❌ 요청 실패 ({response.status_code}): {response.text}")
