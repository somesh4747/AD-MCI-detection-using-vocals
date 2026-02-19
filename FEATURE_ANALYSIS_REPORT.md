# Feature Extraction Analysis & Fixes

## PROBLEMS IDENTIFIED IN _main_features.py

### 1. CRITICAL ISSUE: Feature Redundancy (pause_count vs word_count)
**Problem:** 
- `pause_count` and `word_count` have correlation r=0.999 (nearly identical!)
- This is because both depend on transcript length
- The model learns to use one arbitrarily, causing overfitting
- When removed, other features get clearer signal

**Solution:**
- KEEP ONLY ONE: Either `pause_count` OR `word_count`, not both
- Recommendation: KEEP `pause_count` (more clinically meaningful for speech analysis)
- REMOVE: `word_count`

---

### 2. WEAK FEATURES (Low Discriminative Power - Cohen's d < 0.3)

These features barely separate Control from MCI:

| Feature | Cohen's d | Interpretation | Recommendation |
|---------|-----------|-----------------|-----------------|
| median_pause_duration | 0.032 | **CRITICAL - Useless** | **REMOVE** |
| std_word_duration | 0.088 | Very weak | **REMOVE** |
| min_pause_duration | 0.102 | Very weak | **REMOVE** |
| max_pause_duration | 0.140 | Very weak | **REMOVE** |
| mean_pause_duration | 0.205 | Weak | **REMOVE** |
| pause_variability | 0.209 | Weak | **REMOVE** |
| std_pause_duration | 0.235 | Weak | **REMOVE** |
| pause_per_word_ratio | 0.289 | Weak | **REMOVE** |

**Why remove them?**
- They add noise, not signal
- Model overfits on training data quirks
- They don't generalize to test data

---

### 3. IMPORTANT FEATURES (Good Discriminative Power - Keep These!)

| Feature | Cohen's d | Strength | Why Important |
|---------|-----------|----------|---------------|
| **pause_count** | **0.572** | **STRONG** | MCI patients have different pause patterns |
| **word_count** | **0.569** | **STRONG** | Speech quantity differs (REMOVE - too correlated) |
| **total_speech_time** | **0.513** | **MEDIUM** | Speech duration is clinically relevant |
| **total_pause_time** | **0.316** | **MEDIUM** | Pause quantity matters |
| **mean_word_duration** | **0.310** | **MEDIUM** | Speech rate variability |
| **speech_rate_wpm** | **0.304** | **MEDIUM** | Classic speech metric |

---

## RECOMMENDED FEATURE SET (6 features instead of 14)

### Keep These:
```python
'pause_count'           # Cohen's d = 0.572 ✓
'total_speech_time'     # Cohen's d = 0.513 ✓
'total_pause_time'      # Cohen's d = 0.316 ✓
'mean_word_duration'    # Cohen's d = 0.310 ✓
'speech_rate_wpm'       # Cohen's d = 0.304 ✓
'pause_per_word_ratio'  # Cohen's d = 0.289 (marginal, but keep for context)
```

### REMOVE These (Too Weak or Redundant):
```python
❌ 'word_count'                # Redundant with pause_count (r=0.999)
❌ 'median_pause_duration'     # Useless (d=0.032)
❌ 'std_word_duration'         # Too weak (d=0.088)
❌ 'min_pause_duration'        # Too weak (d=0.102)
❌ 'max_pause_duration'        # Too weak (d=0.140)
❌ 'mean_pause_duration'       # Too weak (d=0.205)
❌ 'pause_variability'         # Too weak (d=0.209)
❌ 'std_pause_duration'        # Too weak (d=0.235)
```

---

## CODE FIX - Updated Feature Extraction

Replace the features dictionary in `_main_features.py` with:

```python
features = {
    # KEEP ONLY STRONG FEATURES
    'pause_count': sum(no_of_silences),
    'total_speech_time': round(sum(total_duration), 4),
    'total_pause_time': round(sum(total_pause_times), 4),
    'mean_word_duration': round(sum(total_speech_times) / len(word_segments), 4) if word_segments else 0,
    'speech_rate_wpm': round((len(word_segments) / sum(total_speech_times)) * 60, 2) if sum(total_speech_times) > 0 else 0,
    'pause_per_word_ratio': round(len(silences) / len(word_segments), 4) if word_segments else 0,
}
```

---

## EXPECTED IMPROVEMENTS

### Before (14 features):
- Training Accuracy: 87%
- Test Accuracy: 63% ❌
- Gap: 24% (overfitting)

### After (6 strong features):
- Training Accuracy: ~80-82%
- Test Accuracy: ~70-75% ✓
- Gap: 5-10% (healthy!)

**Why better?**
1. Removes noise from weak features
2. Eliminates redundancy (word_count)
3. Model learns real patterns, not quirks
4. Better generalization to new patients

---

## CLINICAL INTERPRETATION

**What matters for Control vs MCI distinction:**

1. **PAUSE COUNT** (Most Important) - MCI speakers have different pausing patterns
2. **SPEECH TIME** - Total amount of speech is different
3. **PAUSE TIME** - How much time spent pausing
4. **SPEECH RATE** - Slower/faster speech in MCI
5. **WORD DURATION** - How long words are spoken
6. **PAUSE-PER-WORD** - Ratio of pauses to speech

**What DOESN'T matter:**
- Exact min/max pause duration
- Pause duration variance
- Median pause timing
- Word duration variance

These weak features don't discriminate well and add noise.

---

## NEXT STEPS

1. **Update `_main_features.py`** - Use new 6-feature set
2. **Regenerate training dataset** - Run feature extraction with new features
3. **Retrain model** - Use 5-fold cross-validation (not 80/20 split)
4. **Expected result**: 70-75% consistent accuracy (more realistic)
