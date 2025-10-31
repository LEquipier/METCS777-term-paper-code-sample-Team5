"""
NYC Yellow Taxi Dataset - Supervised Learning Model Comparison Using PySpark

Purpose:
    This script implements a comparative experiment of three supervised learning models:
    - Logistic Regression
    - Support Vector Machine (Linear SVC)
    - Gradient Boosting Trees
    
    The models are trained and evaluated on a sampled NYC Yellow Taxi dataset to predict
    whether fare_amount > 30 (binary classification).

Environment Setup:
    - Requires PySpark 3.x
    - Python 3.7+
    - Java 8 or 11
    
How to Run:
    1. Ensure PySpark is installed: pip install pyspark
    2. Navigate to the term-paper/code directory
    3. Run: python METCS777-term-paper-code-sample-Team5.py

Author: Team 5
Date: 2024-10-31
"""

# =============================================================================
# 1️⃣ ENVIRONMENT SETUP
# =============================================================================

import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

# PySpark ML imports
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, LinearSVC, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def initialize_spark():
    """Initialize SparkSession with optimized configuration for local execution."""
    spark = SparkSession.builder \
        .appName("NYC_Taxi_ML_Comparison") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")  # Reduce verbose logging
    print(f"✅ SparkSession initialized with {spark.sparkContext.defaultParallelism} cores")
    return spark

# =============================================================================
# 2️⃣ DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_preprocess_data(spark, file_path):
    """
    Load NYC taxi dataset and perform preprocessing steps.
    
    Args:
        spark: SparkSession object
        file_path: Path to the CSV file
        
    Returns:
        preprocessed DataFrame with features and target variable
    """
    
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
    
    print("📂 Loading dataset...")
    df = spark.read.csv(file_path, header=False, schema=schema)
    print(f"   Initial dataset size: {df.count():,} records")
    
    # Data cleaning: remove null and invalid records
    print("🧹 Cleaning data...")
    df_clean = df.filter(
        (col("trip_distance") > 0) &
        (col("fare_amount") > 0) &
        (col("pickup_datetime").isNotNull()) &
        (col("trip_distance") < 100) &  # Remove outliers
        (col("fare_amount") < 200)      # Remove outliers
    )
    
    print(f"   After cleaning: {df_clean.count():,} records")
    
    # Convert pickup_datetime to timestamp and extract features
    print("⏰ Extracting temporal features...")
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
    
    # Create target variable: label = 1 if fare_amount > 30 else 0
    df_with_target = df_features.withColumn(
        "label",
        when(col("fare_amount") > 30, 1).otherwise(0)
    )
    
    # Select final feature columns
    feature_cols = ["trip_distance", "passenger_count", "pickup_hour", "pickup_dayofweek"]
    final_df = df_with_target.select(feature_cols + ["label"])
    
    print(f"✨ Preprocessing complete. Features: {feature_cols}")
    print("📊 Target distribution:")
    final_df.groupBy("label").count().show()
    
    return final_df, feature_cols

def prepare_ml_data(df, feature_cols):
    """
    Prepare data for machine learning by assembling features and splitting dataset.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        
    Returns:
        train_df, test_df: Training and test datasets
    """
    
    # Assemble features into a single vector column
    print("🔧 Assembling features...")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_assembled = assembler.transform(df).select("features", "label")
    
    # Split data into training and test sets (80/20)
    print("✂️ Splitting data (80% train, 20% test)...")
    train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)
    
    train_count = train_df.count()
    test_count = test_df.count()
    
    print(f"   Training set: {train_count:,} records")
    print(f"   Test set: {test_count:,} records")
    
    return train_df, test_df

# =============================================================================
# 3️⃣ MODEL TRAINING & EVALUATION
# =============================================================================

def train_logistic_regression(train_df, test_df):
    """Train and evaluate Logistic Regression model."""
    print("\n🤖 Training Logistic Regression...")
    
    start_time = time.time()
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
    lr_model = lr.fit(train_df)
    training_time = time.time() - start_time
    
    # Make predictions
    lr_predictions = lr_model.transform(test_df)
    
    # Evaluate model
    auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    accuracy_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    
    auc = auc_evaluator.evaluate(lr_predictions)
    accuracy = accuracy_evaluator.evaluate(lr_predictions)
    
    print(f"   ✅ Training completed in {training_time:.2f} seconds")
    print(f"   📊 AUC: {auc:.4f}")
    print(f"   🎯 Accuracy: {accuracy:.4f}")
    
    return {
        'model': 'Logistic Regression',
        'auc': auc,
        'accuracy': accuracy,
        'training_time': training_time
    }

def train_svm(train_df, test_df):
    """Train and evaluate Support Vector Machine model."""
    print("\n🤖 Training Support Vector Machine...")
    
    start_time = time.time()
    svm = LinearSVC(featuresCol="features", labelCol="label", maxIter=100)
    svm_model = svm.fit(train_df)
    training_time = time.time() - start_time
    
    # Make predictions
    svm_predictions = svm_model.transform(test_df)
    
    # Evaluate model
    auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    accuracy_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    
    auc = auc_evaluator.evaluate(svm_predictions)
    accuracy = accuracy_evaluator.evaluate(svm_predictions)
    
    print(f"   ✅ Training completed in {training_time:.2f} seconds")
    print(f"   📊 AUC: {auc:.4f}")
    print(f"   🎯 Accuracy: {accuracy:.4f}")
    
    return {
        'model': 'Support Vector Machine',
        'auc': auc,
        'accuracy': accuracy,
        'training_time': training_time
    }

def train_gradient_boosting(train_df, test_df):
    """Train and evaluate Gradient Boosting Trees model."""
    print("\n🤖 Training Gradient Boosting Trees...")
    
    start_time = time.time()
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=20)
    gbt_model = gbt.fit(train_df)
    training_time = time.time() - start_time
    
    # Make predictions
    gbt_predictions = gbt_model.transform(test_df)
    
    # Evaluate model
    auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    accuracy_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    
    auc = auc_evaluator.evaluate(gbt_predictions)
    accuracy = accuracy_evaluator.evaluate(gbt_predictions)
    
    print(f"   ✅ Training completed in {training_time:.2f} seconds")
    print(f"   📊 AUC: {auc:.4f}")
    print(f"   🎯 Accuracy: {accuracy:.4f}")
    
    return {
        'model': 'Gradient Boosting',
        'auc': auc,
        'accuracy': accuracy,
        'training_time': training_time
    }

# =============================================================================
# 4️⃣ RESULTS VISUALIZATION & REPORTING
# =============================================================================

def create_results_table(results):
    """Create and display results comparison table."""
    print("\n" + "="*60)
    print("📈 MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Create DataFrame for results
    results_df = pd.DataFrame(results)
    
    # Print formatted table
    print(f"{'Model':<25} {'AUC':<8} {'Accuracy':<10} {'Training Time (s)':<18}")
    print("-" * 60)
    
    for _, row in results_df.iterrows():
        print(f"{row['model']:<25} {row['auc']:<8.4f} {row['accuracy']:<10.4f} {row['training_time']:<18.2f}")
    
    # Save results to CSV in the results directory
    output_dir = "../results"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'model_comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    return results_df

def plot_results(results_df):
    """Create visualization comparing model performance."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NYC Taxi Dataset - Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = results_df['model']
        
        # AUC comparison
        bars1 = ax1.bar(models, results_df['auc'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Area Under ROC Curve (AUC)')
        ax1.set_ylabel('AUC Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Accuracy comparison
        bars2 = ax2.bar(models, results_df['accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Classification Accuracy')
        ax2.set_ylabel('Accuracy Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Training time comparison
        bars3 = ax3.bar(models, results_df['training_time'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_title('Training Time')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}s', ha='center', va='bottom')
        
        # Combined performance radar chart would go in ax4, but let's use a simple summary
        ax4.axis('off')
        summary_text = "Key Insights:\n\n"
        best_auc = results_df.loc[results_df['auc'].idxmax(), 'model']
        best_acc = results_df.loc[results_df['accuracy'].idxmax(), 'model']
        fastest = results_df.loc[results_df['training_time'].idxmin(), 'model']
        
        summary_text += f"• Highest AUC: {best_auc}\n"
        summary_text += f"• Highest Accuracy: {best_acc}\n"
        summary_text += f"• Fastest Training: {fastest}\n\n"
        summary_text += "Recommendations:\n"
        summary_text += "• For production: Consider accuracy vs speed tradeoff\n"
        summary_text += "• For large datasets: Evaluate scalability\n"
        summary_text += "• For real-time: Prioritize training time"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization to the results directory
        output_dir = "../results"
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, 'model_comparison_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {viz_path}")
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
        print("   (This is normal if running in a headless environment)")

# =============================================================================
# 5️⃣ MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function that orchestrates the entire experiment."""
    
    print("🚕" * 20)
    print("NYC YELLOW TAXI DATASET - ML MODEL COMPARISON")
    print("🚕" * 20)
    
    # Initialize Spark
    spark = initialize_spark()
    
    try:
        # Load and preprocess data from the correct path
        data_path = "../data/taxi-data-top-10k.csv"
        df, feature_cols = load_and_preprocess_data(spark, data_path)
        train_df, test_df = prepare_ml_data(df, feature_cols)
        
        # Train and evaluate all models
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
        
        # Generate results and visualization
        results_df = create_results_table(results)
        plot_results(results_df)
        
        print("\n🎉 Experiment completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   • All three models show different strengths")
        print("   • Consider your specific use case requirements")
        print("   • Scalability matters for production deployment")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up Spark session
        spark.stop()
        print("🛑 SparkSession stopped")

if __name__ == "__main__":
    main()
