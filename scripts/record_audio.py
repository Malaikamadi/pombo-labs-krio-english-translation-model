import sounddevice as sd
from scipy.io.wavfile import write

# Settings
duration = 30 
samplerate = 16000 

print("Recording... Speak now!")
recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()
print("Recording finished!")


write('data/audio/my_test_audio.wav', samplerate, recording)
print("Audio saved to data/audio/my_test_audio.wav")