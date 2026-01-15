# To Eat or Not to Eat? 🍄
## Uncovering Toxicity Patterns in the UCI Mushroom Dataset

### 📌 Project Overview
This project applies Unsupervised Learning techniques, specifically **Association Rule Mining (Apriori Algorithm)**, to identify biological characteristics that determine whether a mushroom is edible or poisonous. 

Unlike standard classification models, this analysis aims to generate explicit, human-readable rules (e.g., *"If Odor is Foul, then Poisonous"*). The goal is to derive a set of "Golden Rules" for safe foraging based on the [UCI Mushroom Data Set](https://archive.ics.uci.edu/dataset/73/mushroom).

### 📂 Project Structure
```text
.
├── data/
│   └── mushrooms_raw.csv       # Original dataset from UCI
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading & cleaning (removing constant cols)
│   ├── map.py                  # Dictionary mapping (single letters -> words)
│   ├── model.py                # Apriori algorithm logic & Rule Pruning
│   └── visualisation.py        # EDA plots & Parallel Coordinates visualization
├── Report.ipynb                # Main analysis notebook (The Report)
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

### ⚙️ Methodology

1.  **Data Preprocessing:**
    * **Mapping:** Converted categorical single-letter codes (e.g., `p`, `e`, `b`) into descriptive labels (e.g., `poisonous`, `edible`, `bell`) using a custom dictionary to ensure readability.
    * **Cleaning:** Identified and removed columns with zero variance (constant values across all observations), as they provide no information gain for association rules.

2.  **Exploratory Data Analysis (EDA):**
    * Visualized the **Class Balance** to ensure the dataset is suitable for rule mining without resampling.
    * Generated a **Correlation Heatmap** to detect multicollinearity between categorical features, identifying highly redundant biological structures.

3.  **Association Rule Mining:**
    * **Algorithm:** Applied the **Apriori Algorithm** (via `mlxtend`) to find frequent itemsets.
    * **Thresholds:** configured strict parameters to filter for high-quality rules:
        * *Min Support:* 0.05 (Pattern must appear in >5% of data - low threshold for generating a lot of rules).
        * *Min Confidence:* 0.6 (Pattern must be correct >60% of the time).
        * *Max Length:* 3 (Limited rule complexity for human interpretability).
    * **Rule Pruning:** Implemented a custom logic to remove redundant rules. If a general rule (e.g., *A -> Poisonous*) exists with equal confidence to a more complex rule (e.g., *A + B -> Poisonous*), the complex version is pruned to reduce noise.

4.  **Visualization:**
    * **Scatter Plot:** Visualized the trade-off between Support and Confidence, mapped with Lift.

### 🚀 Getting Started

#### Prerequisites
* Python 3.8 or higher
* Jupyter Notebook

#### Installation
1.  Clone the repository or download the source files.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Launch the analysis report:
    ```bash
    jupyter notebook Report.ipynb
    ```

### 📊 Key Findings

The analysis successfully isolated **100% confidence rules** that guarantee safety or danger:

* **Primary Indicator (Odor):** Odor is the single most discriminatory feature.
* **Visual Indicators:** In the absence of a distinct odor, specific visual markers act as definitive warnings:
    * **Green Spore Prints** (`spore-print-color=green`) $\rightarrow$ Poisonous.
    * **Buff Gill Color** (`gill-color=buff`) $\rightarrow$ Poisonous.
* **Conclusion:** Reliance on common visual features alone (like cap color) is statistically dangerous due to high overlap between classes. A hierarchical check (Smell First $\rightarrow$ Visuals Second) is the recommended algorithmic approach for foraging.

### 📚 References
* **Dataset Source:** [UCI Machine Learning Repository - Mushroom Data Set](https://archive.ics.uci.edu/ml/datasets/mushroom)

---
*Author: Adam Jaworski*