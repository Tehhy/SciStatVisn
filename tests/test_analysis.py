import pytest
import pandas as pd
import numpy as np

from scistatvisn import (
    select_statistical_test,
    _select_test_for_two_groups,
    _select_test_for_multiple_groups,
    format_posthoc_results,
)


# =====================================================================
# 1. UNIT TESTS FOR HELPER STRATEGY FUNCTIONS
# =====================================================================


def test_select_test_for_two_groups():
    # Paired + Normal + Large Sample = Paired Student's t-test
    res = _select_test_for_two_groups(is_paired=True, is_normal=True, is_small_sample=False)
    assert res["recommended_test"] == "ttest_rel"

    # Paired + Non-normal = Wilcoxon Signed-Rank Test
    res = _select_test_for_two_groups(is_paired=True, is_normal=False, is_small_sample=False)
    assert res["recommended_test"] == "wilcoxon"

    # Independent + Normal + Large Sample = Independent Student's t-test
    res = _select_test_for_two_groups(is_paired=False, is_normal=True, is_small_sample=False)
    assert res["recommended_test"] == "ttest_ind"

    # Independent + Non-normal = Mann-Whitney U Test
    res = _select_test_for_two_groups(is_paired=False, is_normal=False, is_small_sample=False)
    assert res["recommended_test"] == "mannwhitneyu"


def test_select_test_for_multiple_groups():
    # Independent + Normal + Large Sample = One-Way ANOVA + Tukey HSD
    res = _select_test_for_multiple_groups(is_paired=False, is_normal=True, is_small_sample=False)
    assert res["recommended_test"] == "anova_oneway"
    assert res["recommended_post_hoc"] == "tukey_hsd"

    # Independent + Non-normal = Kruskal-Wallis + Dunn's Test
    res = _select_test_for_multiple_groups(is_paired=False, is_normal=False, is_small_sample=False)
    assert res["recommended_test"] == "kruskal_wallis"
    assert res["recommended_post_hoc"] == "dunn_test"

    # Paired + Non-normal = Friedman Test
    res = _select_test_for_multiple_groups(is_paired=True, is_normal=False, is_small_sample=False)
    assert res["recommended_test"] == "friedman"


# =====================================================================
# 2. INTEGRATION TESTS FOR MAIN DISPATCHER FUNCTION
# =====================================================================


@pytest.fixture
def synthetic_data_2_groups_paired():
    """Generates 2 paired normally distributed groups (N=40)."""
    np.random.seed(42)
    n = 40
    subjects = np.repeat(np.arange(1, n + 1), 2)
    conditions = np.tile(["control", "experiment"], n)
    values = np.concatenate([np.random.normal(10, 2, n), np.random.normal(12, 2, n)])
    return pd.DataFrame({"subject_id": subjects, "condition": conditions, "metric_value": values})


@pytest.fixture
def synthetic_data_3_groups_independent():
    """Generates 3 independent non-normally distributed groups."""
    np.random.seed(42)
    n = 35
    df = pd.DataFrame(
        {
            "subject_id": [f"sub_{i}" for i in range(n * 3)],
            "condition": np.repeat(["A", "B", "C"], n),
            "metric_value": np.concatenate(
                [
                    np.random.exponential(scale=2.0, size=n),  # Exponential = Non-normal
                    np.random.exponential(scale=3.0, size=n),
                    np.random.exponential(scale=5.0, size=n),
                ]
            ),
        }
    )
    return df


def test_select_statistical_test_paired_two_groups(synthetic_data_2_groups_paired):
    res = select_statistical_test(synthetic_data_2_groups_paired)
    assert res["actual_num_groups"] == 2
    assert res["is_paired"]
    assert res["recommended_test"] == "ttest_rel"


def test_select_statistical_test_independent_three_groups(synthetic_data_3_groups_independent):
    res = select_statistical_test(synthetic_data_3_groups_independent)
    assert res["actual_num_groups"] == 3
    assert not res["is_paired"]
    assert not res["is_normal"]
    assert res["recommended_test"] == "kruskal_wallis"
    assert res["recommended_post_hoc"] == "dunn_test"


def test_invalid_group_count_mode(synthetic_data_2_groups_paired):
    """Ensures ValueErrors are properly raised when user-enforced mode mismatches dataset."""
    with pytest.raises(
        ValueError, match=r"Mode '3\+' selected, but dataset contains only 2 groups!"
    ):
        select_statistical_test(synthetic_data_2_groups_paired, num_groups_mode="3+")


# =====================================================================
# 3. UNIT TESTS FOR POST-HOC RESULT FORMATTING
# =====================================================================


def test_format_posthoc_results():
    # Create mock square p-value matrix
    matrix_data = {"A": [1.0, 0.04, 0.0001], "B": [0.04, 1.0, 0.25], "C": [0.0001, 0.25, 1.0]}
    posthoc_matrix = pd.DataFrame(matrix_data, index=["A", "B", "C"])

    report = format_posthoc_results(posthoc_matrix)

    # Should output 3 pairwise comparisons: A vs B, A vs C, B vs C
    assert len(report) == 3
    assert "Comparison" in report.columns
    assert "Significance" in report.columns

    # Validate significance annotations
    row_ac = report[report["Comparison"] == "A vs C"].iloc[0]
    assert row_ac["Significance"] == "***"
    assert row_ac["Is Significant"]

    row_bc = report[report["Comparison"] == "B vs C"].iloc[0]
    assert row_bc["Significance"] == "ns"
    assert not row_bc["Is Significant"]
