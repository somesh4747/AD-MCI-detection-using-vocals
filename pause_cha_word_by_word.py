import pylangacq
import pandas as pd
import re


def get_patient_word_segments(file_path):
    """
    Extract all patient (PAR) words with their individual timings from a .cha file.
    Returns a list of word segments that can be used to chop/extract audio word by word.
    Parses the file directly to extract %wor timing lines.
    """
    word_segments = []
    word_count = 0
    par_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total lines in file: {len(lines)}")
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip('\n')
            
            # Look for patient utterance lines
            if line.startswith('*PAR:'):
                par_count += 1
                print(f"\nFound PAR utterance #{par_count} at line {i}: {line[:50]}")
                
                # Look ahead for the %wor: line (may be after %mor: and %gra: lines)
                j = i + 1
                found_wor = False
                
                while j < len(lines) and j < i + 10:  # Look within next 10 lines
                    next_line = lines[j].rstrip('\n')
                    
                    if next_line.startswith('%wor:'):
                        found_wor = True
                        print(f"  Found %wor line!")
                        # Parse the word timings
                        wor_content = next_line.replace('%wor:', '').strip()
                        
                        # Split by spaces to get words and timings
                        parts = wor_content.split()
                        print(f"  Parts count: {len(parts)}")
                        
                        k = 0
                        while k < len(parts):
                            word = parts[k]
                            
                            # Check if next item is a timing (contains underscore)
                            # The timing might have tab characters, so clean it
                            if k + 1 < len(parts):
                                timing_raw = parts[k + 1]
                                # Remove tab characters and other whitespace
                                timing = timing_raw.replace('\x15', '').strip()
                                
                                if '_' in timing:
                                    try:
                                        start_ms, end_ms = map(float, timing.split('_'))
                                        
                                        # Convert milliseconds to seconds
                                        start_sec = start_ms / 1000.0
                                        end_sec = end_ms / 1000.0
                                        duration_sec = end_sec - start_sec
                                        
                                        word_count += 1
                                        
                                        word_segments.append({
                                            'word_num': word_count,
                                            'word': word,
                                            'start_ms': start_ms,
                                            'end_ms': end_ms,
                                            'start_sec': round(start_sec, 3),
                                            'end_sec': round(end_sec, 3),
                                            'duration_sec': round(duration_sec, 3)
                                        })
                                        
                                        k += 2
                                    except ValueError as e:
                                        print(f"    Error parsing timing '{timing}': {e}")
                                        k += 1
                                else:
                                    k += 1
                            else:
                                k += 1
                        
                        break
                    elif next_line.startswith('*'):
                        # Reached next utterance without finding %wor
                        print(f"  Reached next utterance without %wor")
                        break
                    else:
                        j += 1
                
                if not found_wor:
                    print(f"  Warning: No %wor line found for this PAR utterance")
            
            i += 1
        
        print(f"\n\nTotal PAR utterances found: {par_count}")
        print(f"Total words extracted: {word_count}")
    
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
    
    return word_segments


def print_word_segments(segments):
    """Print word-level segments in a readable format"""
    
    print("=" * 110)
    print("PATIENT VOICE SEGMENTS - WORD BY WORD (Ready for Chopping)")
    print("=" * 110)
    print(f"\nTotal Patient Words: {len(segments)}\n")
    
    total_voice_time = 0
    
    for seg in segments:
        total_voice_time += seg['duration_sec']
        print(f"Word {seg['word_num']:4d}: '{seg['word']:20s}' | "
              f"Start: {seg['start_sec']:8.3f}s | End: {seg['end_sec']:8.3f}s | "
              f"Duration: {seg['duration_sec']:6.3f}s")
    
    print("\n" + "=" * 110)
    print(f"Total Patient Voice Time: {total_voice_time:.2f} seconds ({total_voice_time/60:.2f} minutes)")
    print(f"Average Word Duration: {total_voice_time/len(segments):.3f} seconds" if segments else "No segments")
    print("=" * 110)


def save_word_segments(segments, output_file):
    """Save word segments to CSV for easy use with audio processing tools"""
    
    df = pd.DataFrame(segments)
    df.to_csv(output_file, index=False)
    print(f"\nWord segments saved to: {output_file}")
    
    # Also create a script-friendly format for FFmpeg
    script_file = output_file.replace('.csv', '_ffmpeg_commands.txt')
    with open(script_file, 'w') as f:
        f.write("# FFmpeg commands to extract patient voice segments - WORD BY WORD\n")
        f.write("# Usage: ffmpeg -i input.mp3 -ss START -to END -c copy output.mp3\n\n")
        
        for seg in segments:
            f.write(f"# Word {seg['word_num']}: {seg['word']}\n")
            f.write(f"ffmpeg -i input.mp3 -ss {seg['start_sec']} -to {seg['end_sec']} ")
            f.write(f"-c copy word_{seg['word_num']:04d}_{seg['word']}.mp3\n\n")
    
    print(f"FFmpeg commands saved to: {script_file}")


def get_word_segments_as_list(segments):
    """Return segments as a simple list for programmatic use"""
    return [(seg['word'], seg['start_sec'], seg['end_sec']) for seg in segments]


def create_silence_map(segments, total_duration=None):
    """
    Create a map of silence (gaps) between words.
    Useful for identifying pauses within speech.
    """
    if not segments:
        return []
    
    silences = []
    for i in range(len(segments) - 1):
        current_end = segments[i]['end_sec']
        next_start = segments[i + 1]['start_sec']
        
        silence_duration = next_start - current_end
        
        if silence_duration > 0:  # Only if there's a gap
            silences.append({
                'between_word': f"{segments[i]['word']} -> {segments[i + 1]['word']}",
                'silence_start': round(current_end, 3),
                'silence_end': round(next_start, 3),
                'silence_duration_sec': round(silence_duration, 3)
            })
    
    return silences


# Example usage
if __name__ == '__main__':
    file_path = r"E:\ML\silero-python\dematia_bank\Baycrest2103.cha"
    
    # Get all patient word segments
    word_segments = get_patient_word_segments(file_path)
    
    if word_segments:
        # Print analysis
        print_word_segments(word_segments)
        
        # Save to CSV and FFmpeg commands
        output_csv = r"E:\ML\silero-python\patient_word_segments.csv"
        save_word_segments(word_segments, output_csv)
        
        # Get as simple list for programmatic use
        word_list = get_word_segments_as_list(word_segments)
        print(f"\nWord List Format (for chopping):")
        print(f"{'Word':<20} {'Start (s)':<15} {'End (s)':<15}")
        print("-" * 50)
        for word, start, end in word_list[:10]:  # Show first 10 words
            print(f"{word:<20} {start:<15} {end:<15}")
        if len(word_list) > 10:
            print(f"... and {len(word_list) - 10} more words")
        
        # Create silence map
        silences = create_silence_map(word_segments)
        if silences:
            print(f"\n\nSILENCES BETWEEN WORDS (Total: {len(silences)}):")
            print("-" * 80)
            silence_df = pd.DataFrame(silences)
            silence_csv = output_csv.replace('.csv', '_silences.csv')
            silence_df.to_csv(silence_csv, index=False)
            print(f"Silence map saved to: {silence_csv}")
            
            # Show top silences
            silence_df_sorted = silence_df.sort_values('silence_duration_sec', ascending=False)
            print(f"\nTop 10 Longest Silences:")
            print(silence_df_sorted.head(10).to_string(index=False))
    else:
        print("No patient word segments found!")
