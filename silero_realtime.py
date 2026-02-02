import pyaudio
import numpy as np
import torch
from collections import deque
from silero_vad import load_silero_vad, get_speech_timestamps

# Configuration
SAMPLING_RATE = 16000
CHUNK_SIZE = 512
CHECK_INTERVAL = 10  # Check every 10 chunks (~320ms)
BUFFER_SIZE = SAMPLING_RATE * 2  # Keep 2 seconds of audio

# Initialize model
print("Loading Silero VAD model...")
model = load_silero_vad()

# Setup audio stream
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=SAMPLING_RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
)

print("Listening for speech... (Press Ctrl+C to stop)\n")

audio_buffer = deque(maxlen=BUFFER_SIZE)
chunk_count = 0
is_speaking = False

try:
    while True:
        # Read audio chunk from microphone
        audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
        
        # Add to buffer
        audio_buffer.extend(audio_data)
        chunk_count += 1
        
        # Check for speech every CHECK_INTERVAL chunks
        if chunk_count >= CHECK_INTERVAL:
            chunk_count = 0
            
            if len(audio_buffer) > SAMPLING_RATE:  # At least 1 second of audio
                # Convert buffer to tensor
                audio_tensor = torch.tensor(list(audio_buffer)).float()
                # print(audio_tensor)
                # Get speech timestamps
                speech_timestamps = get_speech_timestamps(
                    audio_tensor, 
                    model, 
                    sampling_rate=SAMPLING_RATE
                )
                
                # Check if currently speaking
                if speech_timestamps:
                    # Get the most recent speech segment
                    latest_segment = speech_timestamps[-1]
                    current_time = len(audio_buffer) / SAMPLING_RATE
                    
                    if latest_segment['end'] >= current_time - 0.5:  # Recent speech
                        if not is_speaking:
                            print("🎤 SPEAKING DETECTED")
                            is_speaking = True
                    else:
                        if is_speaking:
                            print("🔇 SILENCE")
                            is_speaking = False
                else:
                    if is_speaking:
                        print("🔇 SILENCE")
                        is_speaking = False

except KeyboardInterrupt:
    print("\n\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Done!")
