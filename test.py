import cv2
import pyttsx3
from rapidfuzz import process

print("========================================")
print("1. Testing OpenCV version:", cv2.__version__)

print("2. Testing Text-to-Speech...")
engine = pyttsx3.init()
engine.setProperty('rate', 150)
print("   TTS Initialized successfully!")

print("3. Testing RapidFuzz...")
choices = ["Doliprane 1000mg", "Amoxicillin 500mg", "Ibuprofen 400mg"]
query = "dolipran"
best_match = process.extractOne(query, choices)
print(f"   Fuzzy match for '{query}': {best_match[0]} (Score: {best_match[1]}%)")

print("========================================")
print(" SUCCESS! All core project tools are ready for Week 1!")
print("========================================")
