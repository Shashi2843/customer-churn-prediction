"""
01_EDA.py - Exploratory Data Analysis notebook
Run this script to perform comprehensive EDA on the customer churn dataset
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set styles
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_explore_data():
    """Load and perform initial exploration"""
    logger.info("=" * 80)
    logger.info("EXPLORATORY DATA ANALYSIS - IBM Telco Customer Churn Dataset")
    logger.info("=" * 80)
    
    from src.data_loader import load_raw_data, download_dataset
    
    # Download dataset
    dataset_path = download_dataset()
    
    # Load data
    df = load_raw_data(dataset_path)
    
    logger.info(f"\nDataset Shape: {df.shape}")
    logger.info(f"\nColumn Names and Types:")
    print(df.dtypes)
    
    logger.info(f"\nFirst 5 rows:")
    print(df.head())
    
    logger.info(f"\nDataset Info:")
    print(df.info())
    
    logger.info(f"\nMissing Values:")
    print(df.isnull().sum())
    
    logger.info(f"\nBasic Statistics:")
    print(df.describe())
    
    return df


def analyze_churn():
    """Analyze churn distribution"""
    logger.info("\n" + "=" * 80)
    logger.info("CHURN ANALYSIS")
    logger.info("=" * 80)
    
    from src.data_loader import load_raw_data, download_dataset
    
    dataset_path = download_dataset()
    df = load_raw_data(dataset_path)
    
    # Churn distribution
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    
    logger.info(f"\nChurn Distribution:")
    logger.info(f"No:  {churn_counts['No']:6d} ({churn_pct['No']:.2f}%)")
    logger.info(f"Yes: {churn_counts['Yes']:6d} ({churn_pct['Yes']:.2f}%)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    churn_counts.plot(kind='bar', ax=axes[0], color=['green', 'red'])
    axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Churn')
    
    # Pie chart
    axes[1].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', colors=['green', 'red'])
    axes[1].set_title('Churn Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("Churn distribution chart saved as 'churn_distribution.png'")
    
    return df


def analyze_customer_demographics(df):
    """Analyze customer demographics"""
    logger.info("\n" + "=" * 80)
    logger.info("CUSTOMER DEMOGRAPHICS ANALYSIS")
    logger.info("=" * 80)
    
    # Tenure analysis
    logger.info(f"\nTenure Statistics (months):")
    logger.info(f"Mean: {df['tenure'].mean():.2f}")
    logger.info(f"Median: {df['tenure'].median():.2f}")
    logger.info(f"Min: {df['tenure'].min()}")
    logger.info(f"Max: {df['tenure'].max()}")
    
    # Monthly charges
    logger.info(f"\nMonthly Charges Statistics:")
    logger.info(f"Mean: ${df['MonthlyCharges'].mean():.2f}")
    logger.info(f"Median: ${df['MonthlyCharges'].median():.2f}")
    logger.info(f"Min: ${df['MonthlyCharges'].min():.2f}")
    logger.info(f"Max: ${df['MonthlyCharges'].max():.2f}")
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Tenure distribution
    df['tenure'].hist(bins=30, ax=axes[0, 0], color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Tenure Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Tenure (months)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Monthly charges distribution
    df['MonthlyCharges'].hist(bins=30, ax=axes[0, 1], color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Monthly Charges Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Monthly Charges ($)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Tenure vs Churn
    df.boxplot(column='tenure', by='Churn', ax=axes[1, 0])
    axes[1, 0].set_title('Tenure by Churn Status', fontweight='bold')
    axes[1, 0].set_xlabel('Churn')
    axes[1, 0].set_ylabel('Tenure (months)')
    
    # Monthly Charges vs Churn
    df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1, 1])
    axes[1, 1].set_title('Monthly Charges by Churn Status', fontweight='bold')
    axes[1, 1].set_xlabel('Churn')
    axes[1, 1].set_ylabel('Monthly Charges ($)')
    
    plt.tight_layout()
    plt.savefig('demographics_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Demographics analysis chart saved as 'demographics_analysis.png'")


def analyze_categorical_features(df):
    """Analyze categorical features"""
    logger.info("\n" + "=" * 80)
    logger.info("CATEGORICAL FEATURES ANALYSIS")
    logger.info("=" * 80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Churn')
    
    logger.info(f"\nCategorical Features ({len(categorical_cols)}):")
    for col in categorical_cols:
        logger.info(f"  - {col}: {df[col].nunique()} unique values")
    
    # Analyze relationship with churn
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols[:6]):
        churn_by_category = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
        churn_by_category.plot(kind='bar', ax=axes[idx], color=['green', 'red'])
        axes[idx].set_title(f'{col} vs Churn (%)', fontweight='bold')
        axes[idx].set_ylabel('Percentage (%)')
        axes[idx].set_xlabel(col)
        axes[idx].legend(['No Churn', 'Churn'])
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Categorical features analysis chart saved as 'categorical_analysis.png'")


def correlation_analysis(df):
    """Analyze feature correlations"""
    logger.info("\n" + "=" * 80)
    logger.info("CORRELATION ANALYSIS")
    logger.info("=" * 80)
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Convert Churn to numeric
    numeric_df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Calculate correlations
    correlations = numeric_df.corr()['Churn'].sort_values(ascending=False)
    
    logger.info(f"\nTop Features Correlated with Churn:")
    for feature, corr in correlations.head(10).items():
        if feature != 'Churn':
            logger.info(f"{feature:25s}: {corr:7.4f}")
    
    # Visualize correlation matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, square=True, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Correlation analysis chart saved as 'correlation_analysis.png'")


def churn_insights(df):
    """Generate churn insights"""
    logger.info("\n" + "=" * 80)
    logger.info("CHURN INSIGHTS & KEY FINDINGS")
    logger.info("=" * 80)
    
    # Customers at high churn risk
    high_churn_risk = df[df['Churn'] == 'Yes']
    low_churn_risk = df[df['Churn'] == 'No']
    
    logger.info(f"\nHigh Churn Risk Customers (n={len(high_churn_risk)}):")
    logger.info(f"  - Avg Tenure: {high_churn_risk['tenure'].mean():.1f} months")
    logger.info(f"  - Avg Monthly Charges: ${high_churn_risk['MonthlyCharges'].mean():.2f}")
    logger.info(f"  - Avg Total Charges: ${high_churn_risk['TotalCharges'].mean():.2f}")
    
    logger.info(f"\nLow Churn Risk Customers (n={len(low_churn_risk)}):")
    logger.info(f"  - Avg Tenure: {low_churn_risk['tenure'].mean():.1f} months")
    logger.info(f"  - Avg Monthly Charges: ${low_churn_risk['MonthlyCharges'].mean():.2f}")
    logger.info(f"  - Avg Total Charges: ${low_churn_risk['TotalCharges'].mean():.2f}")
    
    # Contract type impact
    logger.info(f"\nChurn Rate by Contract Type:")
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    print(contract_churn)
    
    # Internet service impact
    logger.info(f"\nChurn Rate by Internet Service:")
    internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
    print(internet_churn)


def main():
    """Run complete EDA"""
    logger.info("Starting EDA pipeline...\n")
    
    # Load and explore
    df = load_and_explore_data()
    
    # Analyze churn
    analyze_churn()
    
    # Demographics
    analyze_customer_demographics(df)
    
    # Categorical features
    analyze_categorical_features(df)
    
    # Correlations
    correlation_analysis(df)
    
    # Insights
    churn_insights(df)
    
    logger.info("\n" + "=" * 80)
    logger.info("EDA COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nGenerated visualizations:")
    logger.info("  - churn_distribution.png")
    logger.info("  - demographics_analysis.png")
    logger.info("  - categorical_analysis.png")
    logger.info("  - correlation_analysis.png")


if __name__ == "__main__":
    main()
