import pandas as pd
from scipy import stats


def _select_test_for_two_groups(is_paired: bool, is_normal: bool, is_small_sample: bool) -> dict:
    """
    Selects appropriate statistical test for comparing 2 groups.
    """
    if is_paired:
        test = "ttest_rel" if (is_normal and not is_small_sample) else "wilcoxon"
    else:
        test = "ttest_ind" if (is_normal and not is_small_sample) else "mannwhitneyu"

    return {"recommended_test": test, "recommended_post_hoc": None}


def _select_test_for_multiple_groups(
    is_paired: bool, is_normal: bool, is_small_sample: bool
) -> dict:
    """
    Selects appropriate main test and post-hoc test for comparing 3 or more groups.
    """
    if is_paired:
        if is_normal and not is_small_sample:
            test = "rm_anova"
            post_hoc = "paired_ttest_bonferroni"
        else:
            test = "friedman"
            post_hoc = "nemenyi_or_dunn"
    else:
        if is_normal and not is_small_sample:
            test = "anova_oneway"
            post_hoc = "tukey_hsd"
        else:
            test = "kruskal_wallis"
            post_hoc = "dunn_test"

    return {"recommended_test": test, "recommended_post_hoc": post_hoc}


def select_statistical_test(
    df_subject: pd.DataFrame,
    group_col: str = "condition",
    value_col: str = "metric_value",
    subject_col: str = "subject_id",
    num_groups_mode: str = "auto",  # Options: 'auto', '2', or '3+'
    alpha: float = 0.05,
) -> dict:
    """
    Main dispatcher function to analyze data characteristics and select
    the appropriate statistical test.

    Parameters:
    -----------
    df_subject : pd.DataFrame
        Aggregated dataset for analysis.
    group_col : str, default='condition'
        Column name representing the grouping factor/condition.
    value_col : str, default='metric_value'
        Column name representing the continuous metric.
    subject_col : str, default='subject_id'
        Column name representing subject identifiers (used for pairing check).
    num_groups_mode : str, default='auto'
        Selection mode: 'auto' (detects group count automatically),
        '2' (enforces 2 groups logic), or '3+' (enforces 3+ groups logic).
    alpha : float, default=0.05
        Significance threshold for normality tests.

    Returns:
    --------
    dict:
        Summary dictionary containing test recommendations, data properties,
        and sample size statistics.
    """
    clean_df = df_subject.dropna(subset=[group_col, value_col]).copy()
    groups = clean_df[group_col].unique()
    actual_num_groups = len(groups)

    if actual_num_groups < 2:
        raise ValueError("At least 2 unique groups are required for analysis.")

    # 1. Mode determination (2 groups vs 3+)
    if num_groups_mode == "2":
        if actual_num_groups != 2:
            raise ValueError(f"Mode '2' selected, but dataset contains {actual_num_groups} groups!")
        use_multiple_groups = False
    elif num_groups_mode == "3+":
        if actual_num_groups < 3:
            raise ValueError(
                f"Mode '3+' selected, but dataset contains only {actual_num_groups} groups!"
            )
        use_multiple_groups = True
    else:
        # Automatic selection based on actual group count
        use_multiple_groups = actual_num_groups >= 3

    # 2. Assess data characteristics (Pairing, Sample Size)
    grouped_data = [g[value_col].values for _, g in clean_df.groupby(group_col)]
    min_n = min(len(g) for g in grouped_data)

    is_paired = False
    if subject_col in clean_df.columns:
        subject_counts = clean_df.groupby(subject_col)[group_col].nunique()
        valid_subjects = subject_counts[subject_counts == actual_num_groups].index
        if len(valid_subjects) > 0 and len(valid_subjects) == len(clean_df[subject_col].unique()):
            is_paired = True
            clean_df = clean_df[clean_df[subject_col].isin(valid_subjects)]

    n_subjects = clean_df[subject_col].nunique() if is_paired else min_n
    is_small_sample = n_subjects < 30

    # 3. Normality assessment
    is_normal = True
    if is_paired and actual_num_groups == 2:
        # Check normality of differences for paired 2-group comparison
        piv = clean_df.pivot(index=subject_col, columns=group_col, values=value_col).dropna()
        diff = piv.iloc[:, 0] - piv.iloc[:, 1]
        p_norm = stats.shapiro(diff).pvalue if len(diff) >= 3 else 0
        is_normal = p_norm > alpha
    else:
        # Check normality for each individual group
        for g in grouped_data:
            p_val = stats.shapiro(g).pvalue if len(g) >= 3 else 0
            if p_val <= alpha:
                is_normal = False
                break

    # 4. Invoke appropriate test selection strategy
    if use_multiple_groups:
        test_selection = _select_test_for_multiple_groups(is_paired, is_normal, is_small_sample)
    else:
        test_selection = _select_test_for_two_groups(is_paired, is_normal, is_small_sample)

    return {
        "actual_num_groups": actual_num_groups,
        "mode_used": "3+" if use_multiple_groups else "2",
        "recommended_test": test_selection["recommended_test"],
        "recommended_post_hoc": test_selection["recommended_post_hoc"],
        "is_paired": is_paired,
        "is_normal": is_normal,
        "is_small_sample": is_small_sample,
        "min_sample_size": n_subjects,
    }


def format_posthoc_results(posthoc_matrix: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Converts a square p-value post-hoc matrix into a clean pairwise comparison report table.

    Parameters:
    -----------
    posthoc_matrix : pd.DataFrame
        Square p-value matrix (output from sp.posthoc_dunn, sp.posthoc_tukeyhsd, etc.)
    alpha : float, default=0.05
        Significance threshold.

    Returns:
    --------
    pd.DataFrame:
        Formatted summary DataFrame ready for exporting or reporting.
    """
    groups = posthoc_matrix.columns.tolist()
    records = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            p_val = posthoc_matrix.loc[g1, g2]

            # Assign significance markers
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            else:
                stars = "ns"  # not significant

            is_significant = p_val < alpha

            records.append(
                {
                    "Comparison": f"{g1} vs {g2}",
                    "Group 1": g1,
                    "Group 2": g2,
                    "p_value": round(p_val, 4),
                    "p_formatted": "< 0.001" if p_val < 0.001 else f"{p_val:.4f}",
                    "Significance": stars,
                    "Is Significant": is_significant,
                }
            )

    return pd.DataFrame(records)
