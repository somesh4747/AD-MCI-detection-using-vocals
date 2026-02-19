import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def extract_mfcc_features(file_path, n_mfcc=20, max_pad_len=400):
    # 1. Load the audio file (Resample to 22kHz for consistency)
    audio, sr = librosa.load(file_path, res_type='kaiser_fast', sr=22050)
    
    # 2. Extract MFCCs
    # n_mfcc=20 is standard; higher captures more fine-grained "voice quality"
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # 3. Extract Delta (Velocity) and Delta-Delta (Acceleration)
    # These are crucial for detecting the "hesitations" in Alzheimer's speech
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # 4. Concatenate them into a single feature map (Stacking vertically)
    # This creates a 60-feature height image (20 MFCC + 20 Delta + 20 Delta2)
    comprehensive_mfcc = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
    
    # 5. Padding/Truncating (Ensures all images are the same size for the CNN)
    if (comprehensive_mfcc.shape[1] < max_pad_len):
        pad_width = max_pad_len - comprehensive_mfcc.shape[1]
        comprehensive_mfcc = np.pad(comprehensive_mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        comprehensive_mfcc = comprehensive_mfcc[:, :max_pad_len]
    
    return comprehensive_mfcc

# --- Visualization for Debugging ---
def plot_mfcc(mfcc_data):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_data, x_axis='time')
    plt.colorbar()
    plt.title('MFCC + Delta + Delta-Delta Spectrogram')
    plt.tight_layout()
    plt.show()

# Example usage:
# features = extract_mfcc_features('patient_speech_sample.wav')
# plot_mfcc(features)