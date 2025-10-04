# Update Classifier Page to Use Trained Models

## Quick Summary

The classifier page currently references models that aren't trained:
- ❌ BERT - Not trained (removed from optimized script)
- ❌ LSTM - Not trained (not in optimized script)
- ❌ SVM - Removed (too slow)
- ✅ Random Forest - **TRAINED and BEST** (86.93% F1)
- ✅ Gradient Boosting - TRAINED (85.91% F1)
- ✅ Logistic Regression - TRAINED (77.08% F1)
- ✅ Naive Bayes - TRAINED (76.95% F1)
- ✅ Ensemble - TRAINED (83.18% F1)

## Required Changes

### 1. Update MODEL_INFO in Frontend

File: `frontend/src/app/(dashboard)/classifier/page.tsx`

**Current** (lines 88-117):
```typescript
const MODEL_INFO: Record<ClassifierModelType, { label: string; description: string }> = {
  ensemble: {
    label: 'Ensemble (Recommended)',
    description: 'Combines multiple models for highest accuracy (88.4%)',
  },
  bert: { ... },  // ❌ NOT TRAINED
  lstm: { ... },  // ❌ NOT TRAINED
  svm: { ... },   // ❌ REMOVED
  logistic_regression: { ... },
  naive_bayes: { ... },
  random_forest: { ... },
};
```

**Replace with**:
```typescript
const MODEL_INFO: Record<ClassifierModelType, { label: string; description: string }> = {
  random_forest: {
    label: 'Random Forest ⭐ Best Performance',
    description: 'Highest accuracy at 86.93% F1 score, fast predictions (~5ms)',
  },
  gradient_boosting: {
    label: 'Gradient Boosting',
    description: 'Strong accuracy at 85.91% F1 score, slightly slower predictions',
  },
  ensemble: {
    label: 'Voting Ensemble',
    description: 'Combines top 3 models for robust predictions (83.18% F1)',
  },
  logistic_regression: {
    label: 'Logistic Regression',
    description: 'Fast baseline model with 77.08% F1 score',
  },
  naive_bayes: {
    label: 'Naive Bayes',
    description: 'Probabilistic classifier with 76.95% F1 score',
  },
};
```

### 2. Change Default Model

**Current** (line 125):
```typescript
const [selectedModel, setSelectedModel] = useState<ClassifierModelType>('ensemble');
```

**Replace with**:
```typescript
const [selectedModel, setSelectedModel] = useState<ClassifierModelType>('random_forest');
```

### 3. Update Info Cards Performance Text

**Current** (lines 651-654):
```typescript
The Ensemble model achieves 88.4% accuracy by combining BERT, LSTM, and traditional
ML approaches for robust predictions.
```

**Replace with**:
```typescript
The Random Forest model achieves 86.93% F1 score, providing the best balance of
accuracy and speed from our optimized training pipeline.
```

### 4. Update TypeScript Type

File: `frontend/src/types/classifier.ts`

**Find and update**:
```typescript
export type ClassifierModelType =
  | 'ensemble'
  | 'bert'        // ❌ Remove
  | 'lstm'        // ❌ Remove
  | 'svm'         // ❌ Remove
  | 'logistic_regression'
  | 'naive_bayes'
  | 'random_forest'
  | 'gradient_boosting';  // Add if missing
```

**Replace with**:
```typescript
export type ClassifierModelType =
  | 'random_forest'
  | 'gradient_boosting'
  | 'ensemble'
  | 'logistic_regression'
  | 'naive_bayes';
```

## Implementation Script

Save this as a file and run it:

```typescript
// frontend/src/app/(dashboard)/classifier/page.tsx
// ... (keep all imports and components the same)

// UPDATE THIS SECTION (lines 88-117):
const MODEL_INFO: Record<ClassifierModelType, { label: string; description: string }> = {
  random_forest: {
    label: 'Random Forest ⭐ (Recommended)',
    description: 'Best performance: 86.93% F1 score, fast predictions in ~5ms',
  },
  gradient_boosting: {
    label: 'Gradient Boosting',
    description: 'Strong accuracy: 85.91% F1 score with ensemble learning',
  },
  ensemble: {
    label: 'Voting Ensemble',
    description: 'Combines top models: 83.18% F1 score for robust predictions',
  },
  logistic_regression: {
    label: 'Logistic Regression',
    description: 'Fast baseline: 77.08% F1 score with linear classification',
  },
  naive_bayes: {
    label: 'Naive Bayes',
    description: 'Probabilistic: 76.95% F1 score using Bayesian inference',
  },
};

// ... (rest of the component)

// UPDATE DEFAULT MODEL (line ~125):
const [selectedModel, setSelectedModel] = useState<ClassifierModelType>('random_forest');

// ... (rest stays the same)
```

## Visual Changes

Before:
```
Model Selection Dropdown:
├─ Ensemble (Recommended)
├─ BERT-Based                    ❌ Not trained
├─ LSTM Neural Network           ❌ Not trained
├─ Logistic Regression
├─ Naive Bayes
├─ Support Vector Machine        ❌ Removed
└─ Random Forest
```

After:
```
Model Selection Dropdown:
├─ Random Forest ⭐ (Recommended)  ✅ Best! 86.93%
├─ Gradient Boosting              ✅ 85.91%
├─ Voting Ensemble                ✅ 83.18%
├─ Logistic Regression            ✅ 77.08%
└─ Naive Bayes                    ✅ 76.95%
```

## Testing After Update

1. **Start frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

2. **Navigate to classifier**:
   ```
   http://localhost:3000/classifier
   ```

3. **Verify**:
   - ✅ Default model is "Random Forest ⭐"
   - ✅ Only 5 models in dropdown
   - ✅ No BERT, LSTM, or SVM options
   - ✅ Accuracy percentages match training results

4. **Test classification**:
   - Enter test text
   - Select each model
   - Verify predictions work (when backend integrated)

## Backend Integration Note

The frontend will call:
```
POST /api/v1/classifier/predict
{
  "text": "...",
  "model_type": "random_forest"
}
```

The backend must:
1. Load the trained `.pkl` file from `data/models/`
2. Use the same feature extraction as training
3. Return prediction in expected format

See: `docs/CLASSIFIER_INTEGRATION.md` for backend setup.

## Quick Edit Commands

If you want to edit directly:

```bash
# Open the file
code frontend/src/app/(dashboard)/classifier/page.tsx

# Find these sections:
# Line 88-117: MODEL_INFO object
# Line 125: selectedModel default
# Line 651-654: Performance info text

# Update according to the changes above
```

## Summary of Benefits

After this update:

✅ Frontend only shows models that are actually trained
✅ Best model (Random Forest) is default
✅ Accurate performance metrics displayed
✅ No confusion about unavailable models
✅ Matches optimized training pipeline
✅ Users get best experience with fastest, most accurate model

## Performance Users Will See

With Random Forest as default:
- **Accuracy**: 86.93% F1 score
- **Speed**: ~5ms predictions
- **Training**: Trained in 0.26 seconds
- **Reliability**: Consistently high performance

Much better than showing BERT/LSTM that aren't even trained!
