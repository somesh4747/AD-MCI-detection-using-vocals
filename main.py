import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


model = load_silero_vad()
wav = read_audio("E:/ML/pauses/load_dir/new.wav", sampling_rate=16000)

# 2. Get the timestamps of speech segments
speech_timestamps = get_speech_timestamps(
    wav, model, sampling_rate=16000, return_seconds=True
)

# 3. Stitch only the speech segments together (removes silence)
# cleaned_wav = collect_chunks(speech_timestamps, wav)

# # 4. Save the result
# save_audio("cleaned_audio.wav", cleaned_wav, sampling_rate=16000)
print(speech_timestamps)
total_pauses = 0
max_gap = 0
min_gap = float("inf")
actual_activity = 0
for i in range(1, len(speech_timestamps)):
    # print(speech_timestamps[i])
    gap = speech_timestamps[i]["start"] - speech_timestamps[i - 1]["end"]
    max_gap = max(max_gap, gap)
    min_gap = min(min_gap, gap)
    total_pauses += gap
for i in range(len(speech_timestamps)):
    actual_activity += speech_timestamps[i]["end"] - speech_timestamps[i]["start"]

print(total_pauses, min_gap, max_gap, actual_activity)
# cleaned_audio = collect_chunks(speech_timestamps, wav)
# print(len(cleaned_audio))
