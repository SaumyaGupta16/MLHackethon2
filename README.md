# Hackathon Age Group Classification - Final Model

This project uses a cleaned and class-balanced XGBoost classifier to predict the `age_group` (Adult or Senior) based on medical and biometric features.

---

## 📁 Files

- `Train_Data.csv` — Labeled training dataset
- `Test_Data.csv` — Unlabeled test dataset
- `final_submission_no_smote.csv` — Final prediction file for submission

---

## ✅ Steps Performed

### 1. Data Cleaning
- Dropped `SEQN` column (identifier)
- Mapped `age_group`: `'Adult' → 0`, `'Senior' → 1`
- Filled missing values with median
- Applied `log1p` transformation to clip & normalize skewed features:
  - `LBXGLU`, `LBXIN`, `LBXGLT`, `BMXBMI`

### 2. Class Imbalance Handling
- Calculated `scale_pos_weight = (# negative samples / # positive samples)`
- Passed into `XGBoost` to handle imbalance without needing SMOTE

### 3. Model
- Used `XGBoostClassifier` with:
  - 400 trees
  - Learning rate = 0.03
  - `scale_pos_weight` to balance classes
  - Stratified train-validation split

### 4. Evaluation
- Printed validation accuracy + classification report
- Tuned for general performance without overfitting

### 5. Submission
- Final predictions saved to `lastfinal.csv` in correct format

---
