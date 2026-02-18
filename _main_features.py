import pandas as pd
import os
import glob
from pause_cha_word_by_word import create_silence_map, get_patient_word_segments


def extract_features_from_patient(cha_file):
    """
    Extract pause and speech timing features from a single patient's .cha file
    
    Returns: Dictionary with all computed features
    """
    
    # Get word segments
    word_segments = get_patient_word_segments(cha_file)
    
    if not word_segments:
        print(f"Warning: No word segments found in {cha_file}")
        return None
    
    # Get silences (pauses between words)
    silences = create_silence_map(word_segments)
    
    if not silences:
        silences_durations = [0]
    else:
        silences_durations = [s['silence_duration_sec'] for s in silences]
    
    # Extract word durations
    word_durations = [w['duration_sec'] for w in word_segments]
    
    # Calculate comprehensive features
    features = {
        # Pause/Silence features
        'mean_pause_duration': round(sum(silences_durations) / len(silences_durations), 4) if silences_durations else 0,
        'std_pause_duration': round(pd.Series(silences_durations).std(), 4),
        'median_pause_duration': round(pd.Series(silences_durations).median(), 4),
        'max_pause_duration': round(max(silences_durations), 4) if silences_durations else 0,
        'min_pause_duration': round(min(silences_durations), 4) if silences_durations else 0,
        'pause_count': len(silences),
        'total_pause_time': round(sum(silences_durations), 4),
        
        # Speech rate features
        'word_count': len(word_segments),
        'total_speech_time': round(sum(word_durations), 4),
        'mean_word_duration': round(sum(word_durations) / len(word_durations), 4) if word_segments else 0,
        'std_word_duration': round(pd.Series(word_durations).std(), 4),
        'speech_rate_wpm': round((len(word_segments) / sum(word_durations)) * 60, 2) if sum(word_durations) > 0 else 0,
        
        # Pause frequency
        'pause_per_word_ratio': round(len(silences) / len(word_segments), 4) if word_segments else 0,
        'pause_variability': round(pd.Series(silences_durations).var(), 4),
    }
    
    return features


def create_training_dataset(patients_dir, output_csv, label_file):
    """
    Create training dataset from multiple patients
    
    Args:
        patients_dir: Directory containing patient .cha files
        output_csv: Path to save the training CSV
        label_file: CSV file with columns: patient_id, diagnosis (0=Control, 1=MCI, 2=AD)
    """
    
    # Load labels
    print(f"Loading patient labels from: {label_file}")
    labels_df = pd.read_csv(label_file)
    label_dict = dict(zip(labels_df['patient_id'], labels_df['diagnosis']))
    
    print(f"Loaded {len(label_dict)} patient labels")
    print(f"Diagnoses: {set(labels_df['diagnosis'])}")
    
    # Find all .cha files
    cha_files = glob.glob(os.path.join(patients_dir, "*.cha"))
    print(f"\nFound {len(cha_files)} .cha files")
    
    all_features = []
    
    for i, cha_file in enumerate(cha_files):
        patient_id = os.path.basename(cha_file).replace('.cha', '')
        
        print(f"\n[{i+1}/{len(cha_files)}] Processing: {patient_id}")
        
        # Get diagnosis for this patient
        if patient_id not in label_dict:
            print(f"  Warning: No diagnosis found for {patient_id}, skipping...")
            continue
        
        diagnosis = label_dict[patient_id]
        
        # Extract features
        features = extract_features_from_patient(cha_file)
        
        if features is not None:
            features['patient_id'] = patient_id
            features['diagnosis'] = diagnosis
            diagnosis_name = {0: 'Control', 1: 'MCI', 2: 'AD'}
            features['diagnosis_name'] = diagnosis_name.get(diagnosis, 'Unknown')
            all_features.append(features)
            print(f"  ✓ Features extracted ({diagnosis_name.get(diagnosis, 'Unknown')})")
        else:
            print(f"  ✗ Failed to extract features")
    
    # Create DataFrame and save
    df_training = pd.DataFrame(all_features)
    df_training.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"Training dataset saved to: {output_csv}")
    print(f"Total patients: {len(df_training)}")
    print(f"\nDiagnosis distribution:")
    print(df_training['diagnosis_name'].value_counts())
    print(f"{'='*80}")
    
    return df_training


# Example usage
if __name__ == '__main__':
    # Directory with patient .cha files
    patients_dir = r"E:\ML\silero-python\Delaware\MCI"
    
    # CSV file with patient diagnoses (you need to create this)
    # Format: patient_id, diagnosis (0=Control, 1=MCI, 2=AD)
    label_file = r"E:\ML\silero-python\_MCI.csv"
    
    # Output training CSV
    output_csv = r"E:\ML\silero-python\training_dataset_MCI.csv"
    
    # Create training dataset
    df = create_training_dataset(patients_dir, output_csv, label_file)