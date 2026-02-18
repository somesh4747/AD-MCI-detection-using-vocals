import pylangacq
import pandas as pd


def get_patient_voice_segments(file_path):
    """
    Extract all patient (PAR) voice segments from a .cha file with timestamps in seconds.
    Returns a list of segments that can be used to chop/extract audio.
    """
    reader = pylangacq.read_chat(file_path)
    utterances = reader.utterances()
    
    patient_segments = []
    
    for utt in utterances:
        # Only get patient (PAR) utterances
        if utt.participant == "PAR":
            if utt.time_marks:
                start_ms = utt.time_marks[0]
                end_ms = utt.time_marks[1]
                
                # Convert milliseconds to seconds
                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0
                
                duration_sec = end_sec - start_sec
                
                # Get the text from tokens
                try:
                    tokens = utt.tokens()
                    text = ' '.join(tokens) if tokens else ""
                except:
                    # Fallback: try to get the main line as string
                    text = str(utt) if utt else ""
                
                patient_segments.append({
                    'segment_num': len(patient_segments) + 1,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'start_sec': round(start_sec, 3),
                    'end_sec': round(end_sec, 3),
                    'duration_sec': round(duration_sec, 3),
                    'text': text
                })
    
    return patient_segments


def print_patient_segments(segments):
    """Print patient voice segments in a readable format"""
    
    print("=" * 100)
    print("PATIENT VOICE SEGMENTS (Ready for Chopping)")
    print("=" * 100)
    print(f"\nTotal Patient Utterances: {len(segments)}\n")
    
    total_voice_time = 0
    
    for seg in segments:
        total_voice_time += seg['duration_sec']
        print(f"Segment {seg['segment_num']}:")
        print(f"  Start: {seg['start_sec']:8.3f}s  |  End: {seg['end_sec']:8.3f}s  |  Duration: {seg['duration_sec']:6.3f}s")
        print(f"  Text: {seg['text'][:80]}")
        print()
    
    print("=" * 100)
    print(f"Total Patient Voice Time: {total_voice_time:.2f} seconds ({total_voice_time/60:.2f} minutes)")
    print("=" * 100)


def save_segments_for_chopping(segments, output_file):
    """Save segments to CSV for easy use with audio processing tools"""
    
    df = pd.DataFrame(segments)
    df.to_csv(output_file, index=False)
    print(f"\nSegments saved to: {output_file}")
    
    # Also create a script-friendly format
    script_file = output_file.replace('.csv', '_ffmpeg_commands.txt')
    with open(script_file, 'w') as f:
        f.write("# FFmpeg commands to extract patient voice segments\n")
        f.write("# Usage: ffmpeg -i input.mp3 -ss START -to END -c copy output.mp3\n\n")
        
        for seg in segments:
            f.write(f"# Segment {seg['segment_num']}\n")
            f.write(f"ffmpeg -i input.mp3 -ss {seg['start_sec']} -to {seg['end_sec']} ")
            f.write(f"-c copy patient_segment_{seg['segment_num']}.mp3\n\n")
    
    print(f"FFmpeg commands saved to: {script_file}")


def get_segments_as_list(segments):
    """Return segments as a simple list for programmatic use"""
    return [(seg['start_sec'], seg['end_sec']) for seg in segments]


# Example usage
if __name__ == '__main__':
    file_path = r"E:\ML\silero-python\Delaware\MCI\01-1.cha"
    
    # Get all patient voice segments
    segments = get_patient_voice_segments(file_path)
    
    # Print analysis
    print_patient_segments(segments)
    
    # Save to CSV and FFmpeg commands
    output_csv = r"E:\ML\silero-python\patient_segments.csv"
    save_segments_for_chopping(segments, output_csv)
    
    # Get as simple list for programmatic use
    segment_list = get_segments_as_list(segments)
    print(f"\nSegment List Format (for chopping):")
    for i, (start, end) in enumerate(segment_list, 1):
        print(f"  Segment {i}: ({start}s, {end}s)")
