import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as ss

def _add_labels(ax):
    """
    Helper function to safely add count labels 
    to the top of the bars in a barplot.
    """
    for p in ax.patches:
        height = p.get_height()
        if height > 0 and not np.isnan(height):
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        fontsize=11, color='black', 
                        xytext=(0, 3), 
                        textcoords='offset points')

def cramers_v(x, y):
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    Values range from 0 to 1, where 1 indicates perfect association.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    with np.errstate(divide='ignore', invalid='ignore'):
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        if min((kcorr-1), (rcorr-1)) == 0:
            return 0
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def plot_correlation_heatmap(df):
    """
    Plots a heatmap of Cramer's V correlation for all categorical features.
    Helps identify redundant features (multicollinearity).
    """
    columns = df.columns
    n = len(columns)
    corr_matrix = np.zeros((n, n))
    
    # calculate Cramer's V for every pair
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr_matrix[i, j] = cramers_v(df[columns[i]], df[columns[j]])

    corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Categorical Feature Correlation (Cramer's V)", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return corr_df

def plot_class_balance(df):
    """
    Plots the specific distribution of the target variable 'poisonous'.
    Use this at the beginning of the report.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    
    # order the bars by count
    order = df['poisonous'].value_counts().index
    
    ax = sns.countplot(x='poisonous', data=df, palette='viridis', order=order)
    
    plt.title('Target Variable Distribution (Edible vs Poisonous)', fontsize=14)
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # apply labels
    _add_labels(ax)
    
    plt.show()

def plot_feature_distribution(df, column_name, top_n=10):
    """
    Plots the distribution of any categorical feature.
    Useful for checking 'odor', 'cap-shape', etc.
    
    Args:
        df: Dataframe
        column_name: The name of the column to visualize
        top_n: Limits the plot to the top N most frequent categories (prevents unreadable plots)
    """
    plt.figure(figsize=(7, 5))
    
    # calculate value counts to order the bars and limit to top_n
    counts = df[column_name].value_counts().head(top_n)
    
    ax = sns.barplot(x=counts.index, y=counts.values, palette='mako')
    
    plt.title(f'Distribution of Feature: {column_name}', fontsize=14)
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    _add_labels(ax)
    
    plt.tight_layout()
    plt.show()

def plot_rules_scatter(rules_df):
    """
    Plots Support vs Confidence.
    - **Color** is a continuous gradient representing Lift.
    - **Size** also represents Lift.
    Uses pure matplotlib for a smooth colorbar instead of binned legend.
    """
    if rules_df.empty:
        print("No rules to visualize.")
        return

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    x = rules_df['support']
    y = rules_df['confidence']
    c = rules_df['lift']

    # calculate Marker Sizes based on Lift
    min_size, max_size = 50, 350
    min_lift, max_lift = c.min(), c.max()
    
    if max_lift == min_lift:
        s = (min_size + max_size) / 2
    else:
        # normalize lift to 0-1 range -> scale to size range
        s = min_size + (c - min_lift) / (max_lift - min_lift) * (max_size - min_size)

    # Scatter Plot
    sc = ax.scatter(x, y, c=c, s=s, cmap='plasma', alpha=0.7, edgecolors='gray', linewidth=0.5)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Lift (Strength)', rotation=270, labelpad=20, fontsize=12)

    ax.set_title('Association Rules: Support vs Confidence\n(Color & Size indicate Lift)', fontsize=15, pad=15)
    ax.set_xlabel('Support (Frequency)', fontsize=12)
    ax.set_ylabel('Confidence (Reliability)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_xlim(left=max(0, x.min()-0.02))
    ax.set_ylim(bottom=max(0, y.min()-0.02), top=min(1.0, y.max()+0.02))

    plt.tight_layout()
    plt.show()