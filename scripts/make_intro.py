from dotenv import load_dotenv

load_dotenv()
from agents.video.intro import generate_intro_clip

print("saved:", generate_intro_clip())