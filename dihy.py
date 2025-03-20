import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = "songtest.mp3"
y, sr = librosa.load(audio_path)

# Separate harmonics and percussives
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Compute tempo and beat frames on the percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr, hop_length=512)

# Convert frames to seconds
beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

print(f"Detected Tempo: {tempo} BPM")
print("Beat Times (first 10):", beat_times[:10])

# Plot the waveform and overlay beat markers
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.6, label='Original waveform')


for bt in beat_times:
    plt.axvline(x=bt, color='r', linestyle='--', alpha=0.8)

plt.title("Detected Beats (Percussive-based)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("songtestbeat_percussive.png")
plt.show()
