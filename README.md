# K-Means Clustering: Customer Spending Analysis

## Overview
This project applies **K-Means Clustering** to analyze customer spending behavior. The goal is to segment customers into distinct groups based on their spending patterns, helping businesses understand their customer base better and make data-driven decisions.

## Dataset
The dataset used for this project contains customer transaction details, including:
- **Customer ID** (Unique identifier)
- **Annual Income** (Income of the customer)
- **Spending Score** (A metric indicating spending behavior)

## Installation
To run this project, ensure you have Python installed along with the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Implementation Steps
1. **Data Preprocessing**:
   - Load the dataset
   - Handle missing values (if any)
   - Normalize features for better clustering
2. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of income and spending scores
   - Identify potential patterns
3. **Applying K-Means Clustering**:
   - Use the **Elbow Method** to determine the optimal number of clusters
   - Train the K-Means algorithm
   - Assign clusters to customers
4. **Visualization & Insights**:
   - Plot clusters using **Matplotlib** and **Seaborn**
   - Interpret the characteristics of each customer segment

## Usage
Run the script using:
```bash
python kmeans_customer_spending.py
```
After execution, the script will display visualizations and save customer segmentation results.

## Results
The project segments customers into clusters such as:
- **Low spenders with low income**
- **High-income customers with moderate spending**
- **Frequent high spenders**
- **Potential loyal customers**

## Future Enhancements
- Implement **Hierarchical Clustering** for comparison
- Integrate customer demographic data for deeper insights
- Deploy as a web-based dashboard

## Author
Kurt Maxwell Kusterer

