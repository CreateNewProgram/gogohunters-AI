import os, requests
from dotenv import load_dotenv

load_dotenv()
# === 환경설정 ===
TYPECAST_API_KEY = os.getenv("TYPECAST_API_KEY")
TYPECAST_ACTOR_ID = os.getenv("TYPECAST_ACTOR_ID")

print(TYPECAST_API_KEY, TYPECAST_ACTOR_ID)