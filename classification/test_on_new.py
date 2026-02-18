import pandas as pd
import joblib
import os
from extract_patient_features import extract_features_from_patient


def predict_patient_diagnosis(cha_file, model_dir='model_artifacts'):
    """
    Predict diagnosis for a new patient
    
    Args:
        cha_file: Path to patient's .cha file
        model_dir: Directory containing trained model artifacts
    
    Returns:
        Dictionary with prediction and probabilities
    """
    
    # Load model, scaler, and features
    model = joblib.load(os.path.join(model_dir, 'ad_mci_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    feature_cols = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    
    # Extract features from new patient
    print(f"Extracting features from: {cha_file}")
    features = extract_features_from_patient(cha_file)
    
    if features is None:
        print("Error: Could not extract features")
        return None
    
    # Create feature vector in correct order
    X_new = pd.DataFrame([features])[feature_cols]
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    # Predict
    prediction = model.predict(X_new_scaled)[0]
    probabilities = model.predict_proba(X_new_scaled)[0]
    
    # Map to diagnosis names
    diagnosis_names = {0: 'Control', 1: 'MCI', 2: 'AD'}
    
    result = {
        'patient_file': cha_file,
        'predicted_diagnosis': diagnosis_names[prediction],
        'prediction_code': prediction,
        'confidence': round(probabilities[prediction] * 100, 2),
        'probabilities': {
            'Control': round(probabilities[0] * 100, 2),
            'MCI': round(probabilities[1] * 100, 2),
            'AD': round(probabilities[2] * 100, 2)
        }
    }
    
    return result


def batch_predict(cha_directory, model_dir='model_artifacts', output_csv=None):
    """
    Predict diagnoses for all patients in a directory
    
    Args:
        cha_directory: Directory containing .cha files
        model_dir: Directory containing trained model
        output_csv: Optional path to save predictions
    """
    
    import glob
    
    cha_files = glob.glob(os.path.join(cha_directory, "*.cha"))
    print(f"Found {len(cha_files)} .cha files\n")
    
    predictions = []
    
    for i, cha_file in enumerate(cha_files, 1):
        print(f"[{i}/{len(cha_files)}] Predicting: {os.path.basename(cha_file)}")
        result = predict_patient_diagnosis(cha_file, model_dir)
        
        if result:
            predictions.append(result)
            print(f"  → {result['predicted_diagnosis']} (Confidence: {result['confidence']}%)")
            print(f"     Control: {result['probabilities']['Control']}%, "
                  f"MCI: {result['probabilities']['MCI']}%, "
                  f"AD: {result['probabilities']['AD']}%\n")
    
    # Save to CSV if requested
    if output_csv and predictions:
        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_csv(output_csv, index=False)
        print(f"\nPredictions saved to: {output_csv}")
    
    return predictions


# Example usage
if __name__ == '__main__':
    # Predict for a single patient
    cha_file = r"E:\ML\silero-python\dematia_bank\Baycrest2103.cha"
    result = predict_patient_diagnosis(cha_file)
    
    if result:
        print("\n" + "="*80)
        print("PREDICTION RESULT")
        print("="*80)
        print(f"File: {result['patient_file']}")
        print(f"Diagnosis: {result['predicted_diagnosis']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"\nAll probabilities:")
        for diag, prob in result['probabilities'].items():
            print(f"  {diag}: {prob}%")
    
    # Batch predict for all patients
    # predictions = batch_predict(
    #     r"E:\ML\silero-python\dematia_bank",
    #     output_csv=r"E:\ML\silero-python\batch_predictions.csv"
    # )