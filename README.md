# Experimental Data Analysis & Visualization Pipeline

An automated statistical workflow for processing experimental datasets from Excel files (`.xlsx`). Selects the statistical test, computes p-values and effect sizes, and visualizes the results.

---

## Table of Contents
1. [Overview](#overview)
2. [Input Data Requirements](#input-data-requirements)
3. [Installation & Setup](#installation--setup)
4. [Pipeline Architecture & Workflow](#pipeline-architecture--workflow)

---

## Overview

This analysis pipeline automates the downstream statistical evaluation of scientific experiments. It is engineered to:
- **Resist bad formatting**: Dynamically map messy headers, handle comma decimal separators, and convert strings to numerical values seamlessly.
- **Select correct statistical tests**: Evaluate group cardinality ($k$), sample dependency (paired vs. independent), normality (Shapiro-Wilk), and homoscedasticity (Levene's test) via `src/scistatvisn/analysis.py`.
- **Derive standardized effect sizes**: Calculate Cohen's $d_z$, Cohen's $d_s$, Rank-Biserial Correlation $r$, or Wilcoxon $r$ with built-in interpretative thresholds.
- **Visualize complex distributions**: Generate hybrid plots overlaying boxplots, individual raw data points, subject-level trajectories (for repeated measures), and automated $p$-value significance brackets.

---

## Input Data Requirements

The pipeline expects an Excel file (`.xlsx`). To ensure smooth execution, your dataset should contain three primary attributes:

| Standard Name | Supported Aliases / Synonyms | Description | Example |
| :--- | :--- | :--- | :--- |
| **`subject_id`** | `subject`, `id` | Unique identifier per subject/participant | `S01`, `102` |
| **`condition`** | `group` | Experimental treatment group or timepoint | `control`, `test` |
| **`metric_value`** | `metric`, `value`, `val` | Dependent continuous variable | `12.5`, `0.84` |

### Smart Schema Normalization & Positional Fallback
- **Case-Insensitive & Whitespace Agnostic**: Column headers are automatically stripped of whitespace and converted to lower case.
- **Positional Fallback**: If unrecognized column names are supplied, the pipeline will fallback to assigning:
  - Column 1 $\to$ subject_id
  - Column 2 $\to$ condition
  - Column 3 $\to$ metric_value
  
> ⚠️ **Warning**: Positional assignment triggers a standard `UserWarning`. Ensure column ordering is maintained if customized headers are omitted.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Tehhy/SciStatVisn](https://github.com/your-username/your-repo-name.git)
   cd SciStatVisn
   ```

2. **Set up a virtual environment and Install dependencies:**

   ```bash
   uv sync --extra dev
   ```

3. **Prepare your data:** Place your Excel file into the `data/` directory (e.g., `data/data2.xlsx`).

---

## Pipeline Architecture & Workflow

```text
[ Raw Data ] ──> [ Prep & Cleaning ] ──> [ Statistical Testing ] ──> 
[ Post-Hoc Analysis ] ──> [ Visualization ]
```

### Analytical Pipeline Overview

The analytical pipeline consists of 5 main stages:

1. **Data Ingestion & Cleaning**: Loading tabular data and primary validation (`openpyxl`, `pandas`).
2. **Exploratory Data Analysis**: Calculating descriptive statistics and analyzing distributions.
3. **Hypothesis Testing**: Testing statistical hypotheses (`scipy`).
4. **Post-Hoc Analysis**: Post-hoc tests and p-value correction for multiple comparisons (`scikit-posthocs`).
5. **Visualization & Output**: Generating publication-quality plots and saving reports (`seaborn`, `matplotlib`).

> 💡 **Detailed Breakdown**: Complete execution steps, step-by-step explanations for each cell, and interactive examples are located directly inside the notebook `notebooks/analysis.ipynb`.

