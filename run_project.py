import subprocess
import sys

print("========================================")
print("   DRIVER DROWSINESS DETECTION")
print("========================================")
print("Starting Driver Drowsiness Detection System...")
print()

subprocess.run([sys.executable, "drowsiness_detector_ui.py"])