import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import os
import glob


def find_audio_file(base_dir, pattern="*.mp3"):
    """Find audio files in directory"""
    print(f"Searching for audio files in: {base_dir}")
    audio_files = glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)
    if audio_files:
        print(f"Found {len(audio_files)} audio file(s):")
        for f in audio_files:
            print(f"  - {f}")
        return audio_files[0]  # Return first found
    else:
        print(f"No audio files found matching {pattern}")
        return None


def cut_audio_by_segments_librosa(audio_file, segments, output_dir, combine=False):
    """
    Cut audio file into segments using librosa.
    
    Args:
        audio_file: Path to the audio file (mp3, wav, etc.)
        segments: List of segment dicts with start_sec and end_sec
        output_dir: Directory to save output files
        combine: If True, combine all segments into one file
    """
    
    # Verify file exists
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading audio file: {audio_file}")
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)
    except Exception as e:
        print(f"Error loading audio: {e}")
        # Try with a specific sample rate
        print("Trying with sr=22050...")
        y, sr = librosa.load(audio_file, sr=22050)
    
    print(f"Audio loaded. Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")
    
    extracted_segments = []
    
    print(f"\nExtracting {len(segments)} segments...")
    for seg in segments:
        start_sample = int(seg['start_sec'] * sr)
        end_sample = int(seg['end_sec'] * sr)
        
        # Extract audio segment
        audio_segment = y[start_sample:end_sample]
        
        # Save individual segment as WAV
        output_file = os.path.join(output_dir, f"segment_{seg['segment_num']:03d}.wav")
        sf.write(output_file, audio_segment, sr)
        
        extracted_segments.append(audio_segment)
        print(f"  Saved segment {seg['segment_num']:3d}: {seg['start_sec']:8.3f}s - {seg['end_sec']:8.3f}s ({len(audio_segment)/sr:.3f}s)")
    
    # Combine all segments if requested
    if combine:
        print(f"\nCombining all segments into one file...")
        combined_audio = np.concatenate(extracted_segments)
        combined_file = os.path.join(output_dir, "combined_patient_audio.wav")
        sf.write(combined_file, combined_audio, sr)
        print(f"Combined audio saved to: {combined_file}")
        print(f"Combined duration: {len(combined_audio)/sr:.2f}s")
    
    print(f"\nAll segments saved to: {output_dir}")
    return extracted_segments


def cut_audio_from_csv(csv_file, audio_file=None, output_dir=None, combine=False):
    """
    Cut audio file based on segments from CSV file.
    
    Args:
        csv_file: Path to CSV file with segment data (from pause_cha.py)
        audio_file: Path to the audio file (if None, will search for it)
        output_dir: Directory to save output segments
        combine: If True, combine all segments into one file
    """
    
    # Read segments from CSV
    print(f"Reading segments from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    segments = df.to_dict('records')
    print(f"Found {len(segments)} segments")
    
    # Find audio file if not provided
    if not audio_file:
        base_dir = os.path.dirname(csv_file)
        # Try to find audio file in dementia_audio directory
        dementia_audio_dir = os.path.join(os.path.dirname(base_dir), "dementia_audio")
        if os.path.exists(dementia_audio_dir):
            audio_file = find_audio_file(dementia_audio_dir, "*.mp3")
        
        if not audio_file:
            print("Could not find audio file automatically. Please provide the path.")
            return
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at: {audio_file}")
        return
    
    # Set default output directory if not provided
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(csv_file), "patient_audio_segments")
    
    print(f"\nUsing audio file: {audio_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        cut_audio_by_segments_librosa(audio_file, segments, output_dir, combine)
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()


# Example usage
if __name__ == '__main__':
    # Paths
    csv_file = r"E:\ML\silero-python\patient_segments.csv"
    
    # Leave audio_file as None to auto-detect, or provide the path:
    audio_file = r"E:\ML\silero-python\dematia_bank\Baycrest2103.mp3"  # Will search in dementia_audio directory
    
    output_dir = r"E:\ML\silero-python\patient_audio_segments"
    
    # Cut audio by segments and optionally combine
    cut_audio_from_csv(
        csv_file=csv_file,
        audio_file=audio_file,
        output_dir=output_dir,
        combine=True  # Set to True to also create a combined file with all patient audio
    )
    
    print("\n" + "=" * 80)
    print("Audio cutting complete!")
    print("=" * 80)
