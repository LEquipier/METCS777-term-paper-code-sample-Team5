# Demo Visualization Script - User Guide

## Overview

This enhanced demonstration script (`demo_visualization.py`) provides comprehensive visualizations and detailed progress tracking for classroom presentations of the NYC Taxi ML models project.

**Created by:** Team 5 (Boran Liu & Hongkai Dai)  
**Course:** METCS777 - Big Data Analytics  
**Purpose:** Interactive classroom demonstration with enhanced visualizations

## Features

### 1. **Real-time Training Progress**
- Step-by-step progress indicators
- Training time tracking for each model
- Clear visual separation of different stages

### 2. **ROC Curves**
- Receiver Operating Characteristic curves for all three models
- Side-by-side comparison
- AUC scores displayed on the plot
- Saved as: `roc_curves_comparison.png`

### 3. **Confusion Matrices**
- Visual confusion matrices for each model
- Color-coded for easy interpretation
- Shows true/false positives and negatives
- Saved as: `confusion_matrices.png`

### 4. **Performance Comparison Charts**
- Three-panel comparison chart:
  - AUC Score Comparison
  - Accuracy Comparison
  - Training Time Comparison
- Saved as: `performance_comparison_demo.png`

### 5. **Comprehensive Summary**
- Final results table
- Key insights highlighting best performers
- Recommendations for different use cases

## How to Run

### Prerequisites

Make sure you have all required packages installed:

```bash
pip install pyspark pandas matplotlib numpy scikit-learn
```

**Note:** `scikit-learn` is required for ROC curve generation.

### Execution

1. **Navigate to the code directory:**
   ```bash
   cd term-paper/code/
   ```

2. **Run the demo script:**
   ```bash
   python demo_visualization.py
   ```

3. **Wait for completion** (~15-20 seconds)
   - The script will display progress for each step
   - Visualizations will be generated automatically
   - Results will be saved to `../results/` directory

## Output Files

The script generates the following files in the `../results/` directory:

| File | Description | Size |
|------|-------------|------|
| `roc_curves_comparison.png` | ROC curves for all three models | ~224 KB |
| `confusion_matrices.png` | Confusion matrices visualization | ~92 KB |
| `performance_comparison_demo.png` | Performance metrics comparison | ~244 KB |
| `model_comparison_results.csv` | Raw numerical results | ~263 B |

## Demo Structure

The demo is organized into three main steps:

### **STEP 1: Data Loading & Preprocessing**
- Load dataset from CSV
- Clean data (remove outliers and nulls)
- Engineer temporal features
- Create target variable
- Display data distribution
- Split into train/test sets

**Key Output:**
```
Initial dataset size: 10,001 records
After cleaning: 9,926 records
Training set: 8,020 samples (80%)
Test set: 1,906 samples (20%)
```

### **STEP 2: Model Training & Evaluation**
For each of the three models:
- Display training progress
- Measure training time
- Calculate AUC and Accuracy
- Show results immediately

**Models Trained:**
1. Logistic Regression
2. Support Vector Machine (LinearSVC)
3. Gradient Boosting Trees

### **STEP 3: Generating Visualizations**
- Generate ROC curves
- Create confusion matrices
- Plot performance comparisons
- Display final summary

## Presentation Tips

### For Classroom Demo:

1. **Introduction** (2 min)
   - Explain the problem: predicting high taxi fares (>$30)
   - Show the feature set: trip distance, passenger count, time features
   - Mention class imbalance: 1.6% positive vs 98.4% negative

2. **Data Preprocessing** (3 min)
   - Show the initial dataset size
   - Explain data cleaning steps
   - Highlight feature engineering
   - Display class distribution

3. **Model Training** (5 min)
   - Run through each model training
   - Point out training time differences
   - Compare AUC and accuracy metrics
   - Discuss why linear models perform well

4. **Visualizations** (5 min)
   - **ROC Curves:** Explain what AUC means, why SVM has the highest
   - **Confusion Matrices:** Show prediction accuracy, discuss false positives/negatives
   - **Performance Charts:** Highlight trade-offs between speed and accuracy

5. **Conclusions** (2 min)
   - Summarize best model for each scenario
   - Discuss scalability with Spark
   - Answer questions

## Key Insights to Highlight

### 1. **Model Performance**
- **Best AUC:** Support Vector Machine (0.9718)
- **Best Accuracy:** Logistic Regression (0.9942)
- **Fastest Training:** Logistic Regression (~1.9s)

### 2. **Class Imbalance Handling**
- Despite 98.4% negative class, all models perform well
- High AUC scores show good discrimination ability
- Confusion matrices show minimal false positives

### 3. **Speed vs. Accuracy Trade-off**
- Logistic Regression: Best balance
- SVM: Highest quality, 2.8x slower
- Gradient Boosting: Moderate on both metrics

### 4. **Practical Applications**
- **Production:** Use Logistic Regression for speed
- **Quality-Critical:** Use SVM for best predictions
- **Complex Patterns:** Use Gradient Boosting for non-linear relationships

## Differences from Main Script

This demo script differs from the main `METCS777-term-paper-code-sample-Team5.py` in the following ways:

| Feature | Main Script | Demo Script |
|---------|-------------|-------------|
| **ROC Curves** | ❌ Not included | ✅ Full ROC visualization |
| **Confusion Matrices** | ❌ Not included | ✅ Side-by-side matrices |
| **Progress Display** | Basic | ✅ Detailed with stages |
| **Logging Level** | WARN | ERROR (cleaner output) |
| **Output Format** | Compact | ✅ Presentation-friendly |
| **Summary** | Simple table | ✅ Comprehensive insights |

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sklearn'"
**Solution:** Install scikit-learn
```bash
pip install scikit-learn
```

### Issue: Visualizations not displaying
**Solution:** If running in headless environment (e.g., server), images are still saved to `../results/` directory. You can view them after copying to your local machine.

### Issue: "Error loading data"
**Solution:** Make sure you're running from the `term-paper/code/` directory and that `../data/taxi-data-top-10k.csv` exists.

### Issue: Spark warnings during execution
**Solution:** These are normal. The demo sets log level to ERROR to minimize noise. Core warnings are expected and don't affect results.

## Customization

To modify the demo for your needs:

1. **Change data path:** Edit line 104
   ```python
   data_path = "../data/taxi-data-top-10k.csv"
   ```

2. **Adjust train/test split:** Edit line 163
   ```python
   train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)
   ```

3. **Modify visualization colors:** Edit lines 296, 398, 464
   ```python
   colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
   ```

4. **Change model parameters:** Edit lines 237, 243, 249 in `main_demo()`

## Questions?

For questions or issues with this demo script, please contact:
- **Boran Liu:** jksliu@bu.edu
- **Hongkai Dai:** daihk@bu.edu

---

**Happy Presenting!** 🎓
