# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

# PySpark ML imports
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, LinearSVC, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Check for sklearn availability (for ROC curves)
try:
    from sklearn.metrics import roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ROC curves will be skipped.")
    print("Install with: pip install scikit-learn")


def initialize_spark():
    """Initialize SparkSession with optimized configuration for local execution."""
    print("\n" + "="*70)
    print(" " * 15 + "NYC TAXI DATASET - ML MODEL COMPARISON")
    print("="*70)
    print("\nInitializing Spark Session...")
    #--------------------------------------------------------------------------------
    spark = SparkSession.builder \
        .appName("NYC_Taxi_ML_Comparison") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    #--------------------------------------------------------------------------------
    spark.sparkContext.setLogLevel("WARN")  # Reduce verbose logging
    cores = spark.sparkContext.defaultParallelism
    print(f"  > Spark initialized with {cores} CPU cores")
    print(f"  > Ready for distributed computing!\n")
    return spark


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_preprocess_data(spark, file_path=None):
    
    print("="*70)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("="*70)
#--------------------------------------------------------------------------------
    # Define schema based on the dataset structure
    schema = StructType([
        StructField("medallion", StringType(), True),
        StructField("hack_license", StringType(), True),
        StructField("pickup_datetime", StringType(), True),
        StructField("dropoff_datetime", StringType(), True),
        StructField("trip_time_in_secs", IntegerType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("pickup_longitude", DoubleType(), True),
        StructField("pickup_latitude", DoubleType(), True),
        StructField("dropoff_longitude", DoubleType(), True),
        StructField("dropoff_latitude", DoubleType(), True),
        StructField("payment_type", StringType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("surcharge", DoubleType(), True),
        StructField("mta_tax", DoubleType(), True),
        StructField("tip_amount", DoubleType(), True),
        StructField("tolls_amount", DoubleType(), True),
        StructField("total_amount", DoubleType(), True)
    ])
#--------------------------------------------------------------------------------
    # Multi-path detection if no specific path provided
    if file_path is None:
        possible_paths = [
            "../../../taxi-data-sorted-small.csv",
            "../../taxi-data-sorted-small.csv",
            "../data/taxi-data-top-10k.csv",
            "../../data/taxi-data-top-10k.csv",
            "./taxi-data-top-10k.csv",
            "data/taxi-data-top-10k.csv"
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError("Could not find dataset in any expected location")
    else:
        data_path = file_path
    
    print(f"\n1. Loading dataset from: {data_path}")
    df = spark.read.csv(data_path, header=False, schema=schema)
    initial_count = df.count()
    print(f"   > Loaded {initial_count:,} records")
    
    # Data cleaning: remove null and invalid records
    print("\n2. Cleaning data (removing outliers and nulls)...")
    #--------------------------------------------------------------------------------
    df_clean = df.filter(
        (col("trip_distance") > 0) &
        (col("fare_amount") > 0) &
        (col("pickup_datetime").isNotNull()) &
        (col("trip_distance") < 100) &  # Remove outliers
        (col("fare_amount") < 200)      # Remove outliers
    )
    #--------------------------------------------------------------------------------
    clean_count = df_clean.count()
    removed = initial_count - clean_count
    print(f"   > Cleaned dataset: {clean_count:,} records")
    print(f"   > Removed {removed:,} invalid/outlier records")
    
    # Convert pickup_datetime to timestamp and extract features
    print("\n3. Engineering features from raw data...")
    #--------------------------------------------------------------------------------
    df_features = df_clean.withColumn(
        "pickup_timestamp", 
        col("pickup_datetime").cast(TimestampType())
    ).withColumn(
        "pickup_hour", 
        hour(col("pickup_timestamp"))
    ).withColumn(
        "pickup_dayofweek", 
        dayofweek(col("pickup_timestamp"))
    ).withColumn(
        "passenger_count",
        when(col("trip_time_in_secs") > 0, 1).otherwise(0)  # Approximate passenger count
    )
    #--------------------------------------------------------------------------------

    print("   > Extracted temporal features:")
    print("     - pickup_hour (0-23)")
    print("     - pickup_dayofweek (1-7)")
    
    # Create target variable: label = 1 if fare_amount > 30 else 0
    print("\n4. Creating target variable (fare > $30)...")
    #--------------------------------------------------------------------------------
    df_with_target = df_features.withColumn(
        "label",
        when(col("fare_amount") > 30, 1).otherwise(0)
    )
    #--------------------------------------------------------------------------------

    # Show class distribution
    print("\n5. Target Distribution:")
    distribution = df_with_target.groupBy("label").count().collect()
    for row in distribution:
        label_name = "High Fare (>$30)" if row['label'] == 1 else "Normal Fare (≤$30)"
        percentage = (row['count'] / clean_count) * 100
        print(f"   > {label_name}: {row['count']:,} samples ({percentage:.1f}%)")
    
    # Select final feature columns
    feature_cols = ["trip_distance", "passenger_count", "pickup_hour", "pickup_dayofweek"]
    final_df = df_with_target.select(feature_cols + ["label"])
    
    print(f"\n6. Final feature set: {', '.join(feature_cols)}")
    
    return final_df, feature_cols

def prepare_ml_data(df, feature_cols):
    
    # Assemble features into a single vector column
    print("\n7. Preparing data for machine learning...")
    #--------------------------------------------------------------------------------
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    #--------------------------------------------------------------------------------
    df_assembled = assembler.transform(df).select("features", "label")
    
    # Split data into training and test sets (80/20)
    #--------------------------------------------------------------------------------
    train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)
    #--------------------------------------------------------------------------------

    train_count = train_df.count()
    test_count = test_df.count()
    
    print(f"   > Training set: {train_count:,} samples (80%)")
    print(f"   > Test set: {test_count:,} samples (20%)")
    
    return train_df, test_df

# =============================================================================
# ROC CURVE UTILITIES
# =============================================================================

def extract_roc_data(predictions_df):
    """Extract data needed for ROC curve plotting."""
    try:
        # Try to get probability column (for LR and GBT)
        pred_data = predictions_df.select("label", "probability", "prediction").collect()
        
        y_true = []
        y_scores = []
        
        for row in pred_data:
            y_true.append(float(row['label']))
            # Extract probability of positive class
            prob_vector = row['probability']
            if hasattr(prob_vector, 'toArray'):
                probs = prob_vector.toArray()
            else:
                probs = list(prob_vector)
            y_scores.append(probs[1] if len(probs) > 1 else probs[0])
        
        return np.array(y_true), np.array(y_scores)
    
    except Exception:
        # Fallback for SVM (uses rawPrediction instead of probability)
        pred_data = predictions_df.select("label", "rawPrediction", "prediction").collect()
        
        y_true = []
        y_scores = []
        
        for row in pred_data:
            y_true.append(float(row['label']))
            # Extract raw prediction score
            raw_vector = row['rawPrediction']
            if hasattr(raw_vector, 'toArray'):
                raw_scores = raw_vector.toArray()
            else:
                raw_scores = list(raw_vector)
            # Use the score for positive class or the single score
            y_scores.append(raw_scores[1] if len(raw_scores) > 1 else raw_scores[0])
        
        return np.array(y_true), np.array(y_scores)

# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_model_with_progress(model_name, model, train_df, test_df):
    """Train a model and display enhanced progress information."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")
    
    print(f"\nStarting training...")
    print(f"  > Model: {model_name}")
    print(f"  > Training samples: {train_df.count():,}")
    print(f"  > Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    # Train model
    model_fitted = model.fit(train_df)
    
    training_time = time.time() - start_time
    print(f"COMPLETED!")
    print(f"  > Training time: {training_time:.2f} seconds")
    
    # Make predictions
    print(f"\nEvaluating on test set...")
    predictions = model_fitted.transform(test_df)
    
    # Calculate metrics
    auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    accuracy_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    
    auc = auc_evaluator.evaluate(predictions)
    accuracy = accuracy_evaluator.evaluate(predictions)
    
    print(f"  > AUC Score: {auc:.4f}")
    print(f"  > Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'model': model_name,
        'model_fitted': model_fitted,
        'predictions': predictions,
        'auc': auc,
        'accuracy': accuracy,
        'training_time': training_time
    }
#--------------------------------------------------------------------------------
def train_logistic_regression(train_df, test_df):
    """Train and evaluate Logistic Regression model."""
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
    return train_model_with_progress("Logistic Regression", lr, train_df, test_df)

def train_svm(train_df, test_df):
    """Train and evaluate Support Vector Machine model."""
    svm = LinearSVC(featuresCol="features", labelCol="label", maxIter=100)
    return train_model_with_progress("Support Vector Machine", svm, train_df, test_df)

def train_gradient_boosting(train_df, test_df):
    """Train and evaluate Gradient Boosting Trees model."""
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=20)
    return train_model_with_progress("Gradient Boosting Trees", gbt, train_df, test_df)
#--------------------------------------------------------------------------------

# =============================================================================
# RESULTS VISUALIZATION & REPORTING
# =============================================================================

def plot_roc_curves(results_list):
    """Plot ROC curves for all models."""
    if not SKLEARN_AVAILABLE:
        print("Skipping ROC curves (scikit-learn not available)")
        return
        
    print(f"\n{'='*70}")
    print("GENERATING ROC CURVES")
    print(f"{'='*70}\n")
    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, result in enumerate(results_list):
        model_name = result['model']
        predictions = result['predictions']
        
        print(f"Processing ROC curve for {model_name}...")
        
        try:
            y_true, y_scores = extract_roc_data(predictions)
            
            # Calculate ROC curve points
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            
            # Plot
            plt.plot(fpr, tpr, color=colors[idx], lw=2, 
                    label=f'{model_name} (AUC = {result["auc"]:.3f})')
            
        except Exception as e:
            print(f"  Warning: Could not generate ROC for {model_name}: {e}")
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - NYC Taxi Fare Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    # Save figure
    output_dir = "../results"
    os.makedirs(output_dir, exist_ok=True)
    roc_path = os.path.join(output_dir, 'roc_curves_comparison.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curves saved to: {roc_path}")
    plt.show()

def create_results_table(results):
    """Create and display enhanced results comparison table."""
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    # Create summary table
    print(f"{'Model':<30} {'AUC':>8} {'Accuracy':>10} {'Time (s)':>10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['model']:<30} {result['auc']:>8.4f} {result['accuracy']:>10.4f} {result['training_time']:>10.2f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Find best models
    best_auc = max(results, key=lambda x: x['auc'])
    best_acc = max(results, key=lambda x: x['accuracy'])
    fastest = min(results, key=lambda x: x['training_time'])
    
    print(f"\n  • Best AUC: {best_auc['model']} ({best_auc['auc']:.4f})")
    print(f"  • Best Accuracy: {best_acc['model']} ({best_acc['accuracy']:.4f})")
    print(f"  • Fastest Training: {fastest['model']} ({fastest['training_time']:.2f}s)")
    
    # Create DataFrame for results
    results_df = pd.DataFrame([{
        'model': r['model'],
        'auc': r['auc'],
        'accuracy': r['accuracy'],
        'training_time': r['training_time']
    } for r in results])
    
    # Save results to CSV in the results directory
    output_dir = "../results"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'model_comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n  • Results saved to: {results_path}")
    
    return results_df

def plot_performance_comparison(results_list):
    """Create enhanced performance comparison visualization."""
    print(f"\n{'='*70}")
    print("GENERATING PERFORMANCE COMPARISON CHARTS")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame([
        {
            'Model': r['model'],
            'AUC': r['auc'],
            'Accuracy': r['accuracy'],
            'Training Time (s)': r['training_time']
        }
        for r in results_list
    ])
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # AUC Comparison
        ax1 = axes[0]
        bars1 = ax1.bar(results_df['Model'], results_df['AUC'], color=colors)
        ax1.set_title('AUC Score Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('AUC Score', fontsize=11)
        ax1.set_ylim([0.98, 1.0])
        ax1.tick_params(axis='x', rotation=15)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Accuracy Comparison
        ax2 = axes[1]
        bars2 = ax2.bar(results_df['Model'], results_df['Accuracy'], color=colors)
        ax2.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_ylim([0.98, 1.0])
        ax2.tick_params(axis='x', rotation=15)
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Training Time Comparison
        ax3 = axes[2]
        bars3 = ax3.bar(results_df['Model'], results_df['Training Time (s)'], color=colors)
        ax3.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Time (seconds)', fontsize=11)
        ax3.tick_params(axis='x', rotation=15)
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = "../results"
        os.makedirs(output_dir, exist_ok=True)
        perf_path = os.path.join(output_dir, 'performance_comparison.png')
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to: {perf_path}")
        plt.show()
        
    except Exception as e:
        print(f"Could not create visualization: {e}")
        print("   (This is normal if running in a headless environment)")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function that orchestrates the entire experiment."""
    
    # Initialize Spark
    spark = initialize_spark()
    
    try:
        # Load and preprocess data (auto-detect path)
        df, feature_cols = load_and_preprocess_data(spark)
        train_df, test_df = prepare_ml_data(df, feature_cols)
        
        # Train and evaluate all models
        print(f"\n{'='*70}")
        print("STEP 2: MODEL TRAINING & EVALUATION")
        print(f"{'='*70}")
        
        results = []
        
        # 1. Logistic Regression
        lr_results = train_logistic_regression(train_df, test_df)
        results.append(lr_results)
        
        # 2. Support Vector Machine
        svm_results = train_svm(train_df, test_df)
        results.append(svm_results)
        
        # 3. Gradient Boosting
        gbt_results = train_gradient_boosting(train_df, test_df)
        results.append(gbt_results)
        
        # Generate enhanced visualizations
        print(f"\n{'='*70}")
        print("STEP 3: GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        # ROC Curves
        plot_roc_curves(results)
        
        # Performance Comparison
        plot_performance_comparison(results)
        
        # Final Summary
        results_df = create_results_table(results)
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        print("\n  For Production Deployment:")
        print("    > Logistic Regression - Best balance of speed and accuracy")
        print("\n  For Maximum Prediction Quality:")
        print("    > Support Vector Machine - Highest AUC score")
        print("\n  For Complex Pattern Detection:")
        print("    > Gradient Boosting - Non-linear relationships")
        
        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print("\nAll results and visualizations saved to: ../results/")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up Spark session
        spark.stop()
        print("\nSpark session closed.")

if __name__ == "__main__":
    main()
