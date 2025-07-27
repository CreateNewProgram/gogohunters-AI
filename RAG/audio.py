from gtts import gTTS
from pydub import AudioSegment

# 사용자가 입력할 텍스트
input_text = "그럼 우학리에서 다른 유물들은 뭐가 있어?"

# gTTS를 사용하여 텍스트를 음성으로 변환 (언어는 한국어로 설정)
tts = gTTS(text=input_text, lang='ko')

# mp3 파일로 저장
mp3_filename = "output.mp3"
tts.save(mp3_filename)

# mp3 파일을 wav 파일로 변환
sound = AudioSegment.from_mp3(mp3_filename)
wav_filename = "output.wav"
sound.export(wav_filename, format="wav")

print(f"WAV 파일로 저장되었습니다: {wav_filename}")