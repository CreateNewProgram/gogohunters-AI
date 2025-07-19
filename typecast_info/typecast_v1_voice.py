import requests
import os
from dotenv import load_dotenv

load_dotenv()
V1_TYPECAST_API_KEY = os.getenv("V1_TYPECAST_API_KEY")

url = "https://api.typecast.ai/v1/voices"
headers = {"X-API-KEY": V1_TYPECAST_API_KEY}
params = {"model": "ssfm-v21"}  # 선택사항

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    voices = response.json()
    # tc_로 시작하는 voice ID만 필터링
    tc_voices = [voice for voice in voices if voice['voice_id'].startswith('tc_')]
    
    print(f"Found {len(tc_voices)} Typecast voices:")
    for voice in tc_voices:
        print(f"ID: {voice['voice_id']}, Name: {voice['voice_name']}, Emotions: {', '.join(voice['emotions'])}")
else:
    print(f"Error: {response.status_code} - {response.text}")