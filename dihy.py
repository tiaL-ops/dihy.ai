import librosa
import librosa.display
import matplotlib.pyplot as plt


audio_path = "songtest.mp3"
y, sr = librosa.load(audio_path)


tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(f"Tempo: {tempo}")


plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.vlines(librosa.frames_to_time(beat_frames, sr=sr), ymin=-1, ymax=1, color='r', linestyle='--')
plt.title("Detected Beats")
plt.savefig('songtestbeat.png')
plt.show()
