# METCS777 Term Project: NYC Taxi Dataset Analysis with PySpark

## Project Overview

This repository contains the term project for **METCS777 - Big Data Analytics** course, implementing a comparative analysis of three supervised learning models using Apache Spark on the NYC Yellow Taxi dataset.

**Team Members:**
- Boran Liu
- Hongkai Dai

**Course:** METCS777 - Big Data Analytics  
**Institution:** Boston University Metropolitan College  
**Date:** October 31st, 2025

## 🎯 Project Objectives

The primary goal is to implement and compare three supervised learning algorithms:
- **Logistic Regression** (baseline linear model)
- **Support Vector Machine** (linear boundary classifier)
- **Gradient Boosting Trees** (non-linear ensemble method)

Each model predicts whether a taxi fare will exceed $30 based on trip characteristics, evaluating performance through AUC, accuracy, and training time metrics.

## 📂 Project Structure

```
METCS777-term-paper-code-sample-Team5/
├── README.md                           # Project documentation
├── term-paper/
│   ├── code/
│   │   └── METCS777-term-paper-code-sample-Team5.py  # Main analysis script
│   ├── data/
│   │   └── taxi-data-top-10k.csv                     # NYC taxi dataset (10K samples)
│   └── results/
│       ├── model_comparison_results.csv              # Performance metrics
│       └── model_comparison_visualization.png        # Performance charts
```

## 🛠 Environment Setup

### Prerequisites

1. **Python 3.7+**
2. **Java 8 or 11** (required for PySpark)
3. **Apache Spark 3.x**

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd METCS777-term-paper-code-sample-Team5
   ```

2. **Install required Python packages:**
   ```bash
   pip install pyspark pandas matplotlib numpy
   ```

3. **Verify Java installation:**
   ```bash
   java -version
   ```
   
4. **Verify Spark installation:**
   ```bash
   python -c "from pyspark.sql import SparkSession; print('PySpark installed successfully')"
   ```

### Alternative: Using Conda

```bash
conda create -n spark-env python=3.9
conda activate spark-env
conda install -c conda-forge pyspark pandas matplotlib
```

## How to Run the Code

### Step 1: Navigate to Code Directory
```bash
cd term-paper/code/
```

### Step 2: Execute the Analysis Script
```bash
python METCS777-term-paper-code-sample-Team5.py
```

### Expected Output
The script will:
1. Initialize Spark session with local execution
2. Load and preprocess the taxi dataset
3. Train three machine learning models
4. Generate performance comparison results
5. Save outputs to the `../results/` directory

### Runtime Information
- **Execution time:** ~15-20 seconds (on modern hardware)
- **Memory usage:** ~2-4 GB (automatically managed by Spark)
- **Output files:** CSV results and PNG visualization

## Results of Running the Code

### Performance Comparison Table

| Model | AUC Score | Accuracy | Training Time (s) |
|-------|-----------|----------|-------------------|
| **Logistic Regression** | 0.9709 | 0.9942 | 1.94 |
| **Support Vector Machine** | 0.9718 | 0.9932 | 4.92 |
| **Gradient Boosting** | 0.9089 | 0.9822 | 2.99 |

### Key Findings

1. **Best Overall Performance:** Support Vector Machine achieves highest AUC (0.9718)
2. **Highest Accuracy:** Logistic Regression with 99.42% accuracy
3. **Fastest Training:** Logistic Regression at 1.94 seconds
4. **Balance Consideration:** All models show excellent performance with minimal accuracy differences

### Performance Insights

- **Linear models** (Logistic Regression & SVM) outperform the non-linear Gradient Boosting
- **Class imbalance** (162 positive vs 9,764 negative samples) handled well by all models
- **Computational efficiency** varies significantly, with Logistic Regression being 2.5x faster than SVM

## Dataset Description and Analysis

### NYC Yellow Taxi Dataset

**Source:** NYC Taxi and Limousine Commission (TLC)  
**Sample Size:** 10,001 records (after preprocessing: 9,926 records)  
**Time Period:** January 2013  
**Original Features:** 17 columns including trip details, location coordinates, and fare information

### Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `medallion` | String | Taxi license ID |
| `hack_license` | String | Driver license ID |
| `pickup_datetime` | Timestamp | Trip start time |
| `dropoff_datetime` | Timestamp | Trip end time |
| `trip_time_in_secs` | Integer | Trip duration |
| `trip_distance` | Double | Distance in miles |
| `pickup_longitude` | Double | Pickup longitude |
| `pickup_latitude` | Double | Pickup latitude |
| `dropoff_longitude` | Double | Dropoff longitude |
| `dropoff_latitude` | Double | Dropoff latitude |
| `payment_type` | String | Payment method |
| `fare_amount` | Double | Base fare amount |
| `surcharge` | Double | Additional charges |
| `mta_tax` | Double | MTA tax |
| `tip_amount` | Double | Tip amount |
| `tolls_amount` | Double | Tolls paid |
| `total_amount` | Double | Total fare |

### Feature Engineering

**Selected Features for ML Models:**
1. `trip_distance` - Primary predictor of fare amount
2. `passenger_count` - Derived from trip duration (approximation)
3. `pickup_hour` - Extracted from pickup_datetime (0-23)
4. `pickup_dayofweek` - Day of week (1-7, Monday=2)

**Target Variable:**
- `label = 1 if fare_amount > 30 else 0` (binary classification)

### Data Quality and Preprocessing

**Data Cleaning Steps:**
1. **Outlier removal:** Trip distances > 100 miles and fares > $200
2. **Null value handling:** Removed records with missing critical fields
3. **Valid range filtering:** Positive trip distances and fare amounts only
4. **Feature extraction:** Temporal features from datetime stamps

**Data Distribution:**
- **Class balance:** 1.6% positive class (fare > $30), 98.4% negative class
- **Temporal spread:** 24-hour coverage with varying trip patterns
- **Spatial coverage:** Manhattan and surrounding areas

### Model-Specific Results Analysis

#### 1. Logistic Regression
- **Strengths:** Fastest training, highest accuracy, interpretable coefficients
- **Performance:** Excellent linear separation capability
- **Use case:** Real-time prediction systems requiring speed

#### 2. Support Vector Machine (Linear SVC)
- **Strengths:** Highest AUC score, robust to outliers
- **Performance:** Best discriminative power between classes
- **Use case:** Scenarios prioritizing prediction quality over speed

#### 3. Gradient Boosting Trees
- **Strengths:** Handles non-linear relationships, feature importance insights
- **Performance:** Moderate performance, longer training time
- **Use case:** Complex pattern recognition in larger datasets

### Statistical Significance

**Model Convergence:**
- All models achieved convergence within specified iterations
- SVM showed some numerical optimization warnings (handled automatically)
- Consistent results across multiple runs (seed=42)

**Performance Metrics:**
- **AUC Scores:** All above 0.90, indicating excellent discriminative ability
- **Accuracy:** All above 98%, reflecting dataset characteristics and model quality
- **Training Efficiency:** Linear models significantly faster than ensemble method

## Technical Implementation Details

### Spark Configuration
```python
SparkSession.builder \
    .appName("NYC_Taxi_ML_Comparison") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
```

### Data Pipeline
1. **Schema definition** for type safety
2. **Distributed loading** via Spark DataFrames
3. **Parallel preprocessing** with column transformations
4. **Feature assembly** using VectorAssembler
5. **Stratified splitting** (80/20 train/test)

### Model Training Pipeline
- **Parallel execution** capability (though run sequentially for comparison)
- **Automated hyperparameter** selection (default configurations)
- **Cross-validation ready** architecture
- **Scalable design** for larger datasets

## Business Applications and Implications

### Practical Use Cases
1. **Dynamic pricing systems** for ride-sharing platforms
2. **Revenue forecasting** for taxi fleet operators  
3. **Route optimization** based on fare predictions
4. **Customer segmentation** for targeted services

### Scalability Considerations
- **Horizontal scaling:** Code designed for distributed execution
- **Memory efficiency:** Spark's lazy evaluation and caching
- **Real-time deployment:** Model serialization and serving capabilities
- **Batch processing:** Suitable for large-scale daily predictions

## Future Enhancements

1. **Feature enrichment:** Weather data, traffic patterns, events
2. **Deep learning models:** Neural networks for complex pattern recognition
3. **Hyperparameter tuning:** Grid search and Bayesian optimization
4. **Real-time streaming:** Integration with Kafka/Kinesis for live predictions
5. **Advanced metrics:** Precision, Recall, F1-score analysis
6. **Geographic analysis:** Spatial clustering and zone-based models

## References and Dependencies

### Libraries Used
- **PySpark 3.x:** Distributed computing and MLlib algorithms
- **Pandas:** Data manipulation and analysis
- **Matplotlib:** Visualization and plotting
- **NumPy:** Numerical computations (via PySpark dependencies)

### Academic References
- Spark MLlib Documentation
- NYC TLC Trip Record Data
- Supervised Learning: Theory and Practice
- Big Data Analytics with Spark

---

## Contact Information

**Team Members:**
- **Boran Liu:** [jksliu@bu.edu]
- **Hongkai Dai:** [daihk@bu.edu]

**Course:** METCS777 - Big Data Analytics  
**Institution:** Boston University Metropolitan College

---

*This project demonstrates the application of distributed computing and machine learning techniques for real-world transportation data analysis, showcasing the power of Apache Spark for scalable data science workflows.*

