# utils/stats_utils.py
"""Statistical helpers for pose linear metrics.

- Uses rpy2 -> lmerTest + emmeans for mixed models (if available).
- No session_order requirement. If missing, defaults sensible values.
- Exposes:
    - discover_linear_files(root)
    - load_session_csvs(list_of_paths)
    - run_rpy2_lmer(df, dv, feature_label)
    - build_table_with_emmeans(df, out_tex, figs_dir)
    - barplot_ax(...)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Any
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import textwrap
from collections import defaultdict
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import warnings


# rpy2 optional but we activate conversions if available
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    # activate conversion rules (fix ContextVar warning)
    try:
        pandas2ri.activate()
    except Exception:
        # best-effort; localconverter still used in functions
        pass
    _HAVE_RPY2 = True
except Exception:
    _HAVE_RPY2 = False


# ============================================================================
# DATA PREPARATION
# ============================================================================

def aggregate_across_pcs(df, groupby_cols, metric_cols):
    """
    Aggregate metrics across PCs for each trial/condition.
    
    Parameters
    ----------
    df : DataFrame
        Input data with PC column
    groupby_cols : list
        Columns to group by (e.g., ['Pair', 'Trial', 'Condition'])
    metric_cols : list
        Metric columns to aggregate
        
    Returns
    -------
    DataFrame with aggregated metrics
    """
    agg_dict = {col: 'mean' for col in metric_cols}
    df_agg = df.groupby(groupby_cols, as_index=False).agg(agg_dict)
    return df_agg


def prepare_leader_follower_data(df, metric_prefix='SD', id_cols=['Condition', 'Pair', 'Trial']):
    """
    Convert wide-format leader/follower data to long format for RM-ANOVA.
    
    Parameters
    ----------
    df : DataFrame
        Must contain Leader_X and Follower_X columns where X is metric_prefix
    metric_prefix : str
        Metric name (e.g., 'SD', 'Vel')
    id_cols : list
        Identifying columns
        
    Returns
    -------
    DataFrame in long format with 'Role' and metric columns
    """
    leader_col = f'Leader_{metric_prefix}'
    follower_col = f'Follower_{metric_prefix}'
    
    if leader_col not in df.columns or follower_col not in df.columns:
        raise ValueError(f"Missing {leader_col} or {follower_col} in dataframe")
    
    # Create long format
    df_long = pd.DataFrame()
    
    # Leader rows
    df_leader = df[id_cols + [leader_col]].copy()
    df_leader['Role'] = 'Leader'
    df_leader['Value'] = df_leader[leader_col]
    df_leader = df_leader.drop(columns=[leader_col])
    
    # Follower rows
    df_follower = df[id_cols + [follower_col]].copy()
    df_follower['Role'] = 'Follower'
    df_follower['Value'] = df_follower[follower_col]
    df_follower = df_follower.drop(columns=[follower_col])
    
    # Combine
    df_long = pd.concat([df_leader, df_follower], ignore_index=True)
    
    return df_long


def prepare_dyadic_data(df, metric_col, id_cols=['Condition', 'Pair', 'Trial']):
    """
    Prepare dyadic metrics (CRQA, mdRQA) for one-way RM-ANOVA.
    Averages across trials within each Pair-Condition combination.
    
    Parameters
    ----------
    df : DataFrame
        Input data
    metric_col : str
        Name of metric column
    id_cols : list
        Identifying columns (must include 'Condition' and 'Pair')
        
    Returns
    -------
    DataFrame ready for RM-ANOVA
    """
    # Average across trials within Pair-Condition
    df_avg = df.groupby(['Condition', 'Pair'], as_index=False)[metric_col].mean()
    return df_avg


# ============================================================================
# ANOVA FUNCTIONS
# ============================================================================

def run_rm_anova_2way(df, dv='Value', subject='Pair', within=['Condition', 'Role'],
                      aggregate_func='mean'):
    """
    Run 2-way repeated measures ANOVA using rpy2.

    Parameters
    ----------
    df : DataFrame
        Long format data
    dv : str
        Dependent variable column name
    subject : str
        Subject identifier column
    within : list
        Within-subject factors (length 2)
    aggregate_func : str or callable
        How to aggregate multiple observations per cell ('mean', 'median', etc.)

    Returns
    -------
    dict with keys: 'anova_table', 'effect_sizes', 'summary'
    """
    if not _HAVE_RPY2:
        warnings.warn("rpy2 not available, falling back to statsmodels (may have issues)")
        return _run_rm_anova_2way_statsmodels(df, dv, subject, within, aggregate_func)

    try:
        # Remove any NaN values
        df_clean = df.dropna(subset=[dv, subject] + within).copy()

        # Aggregate multiple observations per cell
        groupby_cols = [subject] + within
        if aggregate_func == 'mean':
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].mean()
        elif aggregate_func == 'median':
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].median()
        elif callable(aggregate_func):
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].agg(aggregate_func)
        else:
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].mean()

        # Use rpy2 for RM-ANOVA
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter

        base = importr('base')
        stats = importr('stats')

        # Convert to R dataframe
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_agg)

        # Assign to R environment
        ro.globalenv['df_anova'] = r_df

        # Fit ANOVA model
        factor1, factor2 = within
        ro.r(f'df_anova${subject} <- factor(df_anova${subject})')
        ro.r(f'df_anova${factor1} <- factor(df_anova${factor1})')
        ro.r(f'df_anova${factor2} <- factor(df_anova${factor2})')

        # Run RM-ANOVA with aov
        formula = f'{dv} ~ {factor1} * {factor2} + Error({subject}/({factor1}*{factor2}))'
        ro.r(f'aov_result <- aov({formula}, data=df_anova)')
        ro.r('aov_summary <- summary(aov_result)')

        # Extract results (simplified for now)
        # Note: Proper extraction would require parsing the Error() structure
        print("RM-ANOVA completed using rpy2")

        return {
            'anova_table': None,
            'effect_sizes': {},
            'summary': {
                'n_subjects': df_agg[subject].nunique(),
                'n_observations': len(df_agg),
                'factors': within,
                'dv': dv,
                'method': 'rpy2'
            },
            'model': None
        }

    except Exception as e:
        warnings.warn(f"rpy2 RM-ANOVA failed: {e}, falling back to statsmodels")
        return _run_rm_anova_2way_statsmodels(df, dv, subject, within, aggregate_func)


def _run_rm_anova_2way_statsmodels(df, dv='Value', subject='Pair',
                                     within=['Condition', 'Role'], aggregate_func='mean'):
    """Fallback implementation using statsmodels with proper aggregation."""
    try:
        # Remove any NaN values
        df_clean = df.dropna(subset=[dv, subject] + within).copy()

        # Aggregate multiple observations per cell
        groupby_cols = [subject] + within
        if aggregate_func == 'mean':
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].mean()
        elif aggregate_func == 'median':
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].median()
        elif callable(aggregate_func):
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].agg(aggregate_func)
        else:
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].mean()

        # Run RM-ANOVA
        aovrm = AnovaRM(df_agg, dv, subject, within=within)
        res = aovrm.fit()

        # Extract results
        anova_table = res.anova_table.copy()

        # Calculate effect sizes (partial eta squared)
        effect_sizes = {}
        for effect in anova_table.index:
            ss_effect = anova_table.loc[effect, 'F Value'] * anova_table.loc[effect, 'Num DF']
            ss_error = anova_table.loc[effect, 'Den DF']
            # Approximate partial eta squared
            partial_eta_sq = ss_effect / (ss_effect + ss_error)
            effect_sizes[effect] = partial_eta_sq

        anova_table['partial_eta_sq'] = [effect_sizes.get(idx, np.nan) for idx in anova_table.index]

        # Create summary
        summary = {
            'n_subjects': df_agg[subject].nunique(),
            'n_observations': len(df_agg),
            'factors': within,
            'dv': dv,
            'method': 'statsmodels'
        }

        return {
            'anova_table': anova_table,
            'effect_sizes': effect_sizes,
            'summary': summary,
            'model': res
        }

    except Exception as e:
        warnings.warn(f"RM-ANOVA failed: {e}")
        return None


def run_rm_anova_1way(df, dv, subject='Pair', within='Condition', aggregate_func='mean'):
    """
    Run 1-way repeated measures ANOVA using rpy2 with proper aggregation.

    Parameters
    ----------
    df : DataFrame
        Long format data
    dv : str
        Dependent variable column name (can be column name or 'Value')
    subject : str
        Subject identifier column
    within : str
        Within-subject factor
    aggregate_func : str or callable
        How to aggregate multiple observations per cell ('mean', 'median', etc.)

    Returns
    -------
    dict with keys: 'anova_table', 'effect_sizes', 'summary'
    """
    if not _HAVE_RPY2:
        warnings.warn("rpy2 not available, falling back to statsmodels")
        return _run_rm_anova_1way_statsmodels(df, dv, subject, within, aggregate_func)

    try:
        # Remove any NaN values
        df_clean = df.dropna(subset=[dv, subject, within]).copy()

        # Aggregate multiple observations per cell
        groupby_cols = [subject, within]
        if aggregate_func == 'mean':
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].mean()
        elif aggregate_func == 'median':
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].median()
        elif callable(aggregate_func):
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].agg(aggregate_func)
        else:
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].mean()

        # Use rpy2 for RM-ANOVA
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter

        stats = importr('stats')
        car = None
        try:
            car = importr('car')
        except:
            pass  # car package optional but recommended

        # Convert to R dataframe
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_agg)

        # Assign to R environment
        ro.globalenv['df_anova'] = r_df

        # Ensure factors
        ro.r(f'df_anova${subject} <- factor(df_anova${subject})')
        ro.r(f'df_anova${within} <- factor(df_anova${within})')

        # Run RM-ANOVA with aov
        formula = f'{dv} ~ {within} + Error({subject}/{within})'
        ro.r(f'aov_result <- aov({formula}, data=df_anova)')
        ro.r('aov_summary <- summary(aov_result)')

        # Extract F, p-value, df from the within-subject error term
        # The structure is: aov_summary[[2]][[1]] for Error(subject/within)
        try:
            ro.r('within_summary <- aov_summary[[2]][[1]]')  # Within-subject effects
            ro.r('f_value <- within_summary[1, "F value"]')
            ro.r('p_value <- within_summary[1, "Pr(>F)"]')
            ro.r('df_num <- within_summary[1, "Df"]')
            ro.r('df_den <- within_summary[2, "Df"]')  # Error term

            f_val = float(ro.r('f_value')[0])
            p_val = float(ro.r('p_value')[0])
            df_num = float(ro.r('df_num')[0])
            df_den = float(ro.r('df_den')[0])

            # Calculate partial eta squared
            ss_effect = f_val * df_num
            partial_eta_sq = ss_effect / (ss_effect + df_den)

            # Create results structure matching statsmodels format
            anova_table = pd.DataFrame({
                'F Value': [f_val],
                'Num DF': [df_num],
                'Den DF': [df_den],
                'Pr > F': [p_val],
                'partial_eta_sq': [partial_eta_sq]
            }, index=[within])

            effect_sizes = {within: partial_eta_sq}

            summary = {
                'n_subjects': df_agg[subject].nunique(),
                'n_observations': len(df_agg),
                'factor': within,
                'dv': dv,
                'method': 'rpy2'
            }

            return {
                'anova_table': anova_table,
                'effect_sizes': effect_sizes,
                'summary': summary,
                'model': None
            }

        except Exception as e:
            warnings.warn(f"Failed to extract rpy2 results: {e}, falling back to statsmodels")
            return _run_rm_anova_1way_statsmodels(df, dv, subject, within, aggregate_func)

    except Exception as e:
        warnings.warn(f"rpy2 RM-ANOVA failed: {e}, falling back to statsmodels")
        return _run_rm_anova_1way_statsmodels(df, dv, subject, within, aggregate_func)


def _run_rm_anova_1way_statsmodels(df, dv, subject='Pair', within='Condition',
                                     aggregate_func='mean'):
    """Fallback implementation using statsmodels with proper aggregation."""
    try:
        # Remove any NaN values
        df_clean = df.dropna(subset=[dv, subject, within]).copy()

        # Aggregate multiple observations per cell
        groupby_cols = [subject, within]
        if aggregate_func == 'mean':
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].mean()
        elif aggregate_func == 'median':
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].median()
        elif callable(aggregate_func):
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].agg(aggregate_func)
        else:
            df_agg = df_clean.groupby(groupby_cols, as_index=False)[dv].mean()

        # Run RM-ANOVA
        aovrm = AnovaRM(df_agg, dv, subject, within=[within])
        res = aovrm.fit()

        # Extract results
        anova_table = res.anova_table.copy()

        # Calculate effect size (partial eta squared)
        effect_sizes = {}
        for effect in anova_table.index:
            ss_effect = anova_table.loc[effect, 'F Value'] * anova_table.loc[effect, 'Num DF']
            ss_error = anova_table.loc[effect, 'Den DF']
            partial_eta_sq = ss_effect / (ss_effect + ss_error)
            effect_sizes[effect] = partial_eta_sq

        anova_table['partial_eta_sq'] = [effect_sizes.get(idx, np.nan) for idx in anova_table.index]

        # Create summary
        summary = {
            'n_subjects': df_agg[subject].nunique(),
            'n_observations': len(df_agg),
            'factor': within,
            'dv': dv,
            'method': 'statsmodels'
        }

        return {
            'anova_table': anova_table,
            'effect_sizes': effect_sizes,
            'summary': summary,
            'model': res
        }

    except Exception as e:
        warnings.warn(f"RM-ANOVA failed: {e}")
        return None


# ============================================================================
# POST-HOC TESTS
# ============================================================================

def bonferroni_pairwise(df, dv, group_col='Condition', alpha=0.05):
    """
    Perform pairwise t-tests with Bonferroni correction.
    
    Parameters
    ----------
    df : DataFrame
        Data with DV and grouping variable
    dv : str
        Dependent variable column
    group_col : str
        Grouping column
    alpha : float
        Significance level
        
    Returns
    -------
    DataFrame with pairwise comparison results
    """
    groups = df[group_col].unique()
    n_comparisons = len(list(combinations(groups, 2)))
    alpha_corrected = alpha / n_comparisons
    
    results = []
    for g1, g2 in combinations(groups, 2):
        data1 = df[df[group_col] == g1][dv].dropna()
        data2 = df[df[group_col] == g2][dv].dropna()
        
        # Paired t-test (assuming repeated measures structure)
        # Note: This is simplified - ideally should account for pairing structure
        if len(data1) == len(data2):
            t_stat, p_val = stats.ttest_rel(data1, data2)
        else:
            t_stat, p_val = stats.ttest_ind(data1, data2)
        
        # Cohen's d
        pooled_std = np.sqrt((data1.var() + data2.var()) / 2)
        cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'Group1': g1,
            'Group2': g2,
            'Mean1': data1.mean(),
            'Mean2': data2.mean(),
            'Diff': data1.mean() - data2.mean(),
            't': t_stat,
            'p': p_val,
            'p_corrected': p_val * n_comparisons,  # Bonferroni
            'Significant': (p_val * n_comparisons) < alpha,
            'Cohens_d': cohens_d
        })
    
    return pd.DataFrame(results)


def tukey_hsd_posthoc(df, dv, group_col='Condition'):
    """
    Perform Tukey HSD post-hoc test.
    
    Parameters
    ----------
    df : DataFrame
        Data
    dv : str
        Dependent variable
    group_col : str
        Grouping variable
        
    Returns
    -------
    DataFrame with Tukey HSD results
    """
    df_clean = df[[dv, group_col]].dropna()
    
    tukey = pairwise_tukeyhsd(endog=df_clean[dv], 
                              groups=df_clean[group_col], 
                              alpha=0.05)
    
    # Convert to DataFrame
    results = pd.DataFrame(data=tukey.summary().data[1:], 
                          columns=tukey.summary().data[0])
    
    return results


# ============================================================================
# PC-SPECIFIC ANALYSES
# ============================================================================

def run_pc_specific_anova(df, metric_col, pc_list=None, subject='Pair', 
                         within='Condition', alpha=0.05):
    """
    Run separate 1-way RM-ANOVAs for each PC.
    
    Parameters
    ----------
    df : DataFrame
        Must have 'PC' column
    metric_col : str
        Metric to analyze
    pc_list : list or None
        List of PCs to analyze (None = all)
    subject : str
        Subject identifier
    within : str
        Within-subject factor
    alpha : float
        Significance level
        
    Returns
    -------
    DataFrame with results for each PC
    """
    if pc_list is None:
        pc_list = sorted(df['PC'].unique())
    
    results = []
    
    for pc in pc_list:
        df_pc = df[df['PC'] == pc].copy()
        
        # Average across trials if needed
        df_pc_avg = df_pc.groupby([subject, within], as_index=False)[metric_col].mean()
        
        res = run_rm_anova_1way(df_pc_avg, dv=metric_col, subject=subject, within=within)
        
        if res is not None:
            anova_row = res['anova_table'].loc[within]
            results.append({
                'PC': pc,
                'F': anova_row['F Value'],
                'p': anova_row['Pr > F'],
                'partial_eta_sq': anova_row['partial_eta_sq'],
                'df_num': anova_row['Num DF'],
                'df_den': anova_row['Den DF'],
                'Significant': anova_row['Pr > F'] < alpha
            })
    
    return pd.DataFrame(results)


# ============================================================================
# CORRELATION ANALYSES
# ============================================================================

def compute_correlations_by_condition(df, var1, var2, condition_col='Condition'):
    """
    Compute correlations between two variables within each condition.
    
    Parameters
    ----------
    df : DataFrame
        Data
    var1, var2 : str
        Variable names
    condition_col : str
        Condition grouping column
        
    Returns
    -------
    DataFrame with correlation results by condition
    """
    results = []
    
    for cond in df[condition_col].unique():
        df_cond = df[df[condition_col] == cond]
        
        # Remove NaNs
        df_clean = df_cond[[var1, var2]].dropna()
        
        if len(df_clean) > 2:
            r, p = stats.pearsonr(df_clean[var1], df_clean[var2])
            
            results.append({
                'Condition': cond,
                'n': len(df_clean),
                'r': r,
                'p': p,
                'r_squared': r**2
            })
    
    return pd.DataFrame(results)


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def compute_descriptive_stats(df, metric_col, groupby_cols=['Condition']):
    """
    Compute descriptive statistics (mean, SD, SEM) by groups.
    
    Parameters
    ----------
    df : DataFrame
        Data
    metric_col : str
        Metric column
    groupby_cols : list
        Columns to group by
        
    Returns
    -------
    DataFrame with descriptive statistics
    """
    def sem(x):
        return x.std(ddof=1) / np.sqrt(len(x))
    
    stats_df = df.groupby(groupby_cols)[metric_col].agg([
        ('n', 'count'),
        ('mean', 'mean'),
        ('sd', 'std'),
        ('sem', sem),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    return stats_df


# ============================================================================
# REPORTING HELPERS
# ============================================================================

def format_anova_report(anova_result, effect_name=None):
    """
    Format ANOVA results for reporting.
    
    Parameters
    ----------
    anova_result : dict
        Result from run_rm_anova_*
    effect_name : str or None
        Specific effect to report (None = first row)
        
    Returns
    -------
    str : Formatted APA-style report
    """
    if anova_result is None:
        return "ANOVA failed"
    
    table = anova_result['anova_table']
    
    if effect_name is None:
        effect_name = table.index[0]
    
    row = table.loc[effect_name]
    
    f_val = row['F Value']
    df_num = int(row['Num DF'])
    df_den = int(row['Den DF'])
    p_val = row['Pr > F']
    eta_sq = row['partial_eta_sq']
    
    # Format p-value
    if p_val < 0.001:
        p_str = "p < .001"
    else:
        p_str = f"p = {p_val:.3f}"
    
    report = f"F({df_num}, {df_den}) = {f_val:.2f}, {p_str}, η²p = {eta_sq:.3f}"
    
    return report


def format_posthoc_report(posthoc_df, comparison_cols=('Group1', 'Group2')):
    """
    Format post-hoc comparison results for reporting.
    
    Parameters
    ----------
    posthoc_df : DataFrame
        Results from bonferroni_pairwise or tukey_hsd_posthoc
        
    Returns
    -------
    list of str : Formatted comparison reports
    """
    reports = []
    
    for _, row in posthoc_df.iterrows():
        g1 = row[comparison_cols[0]]
        g2 = row[comparison_cols[1]]
        
        if 'p_corrected' in row:
            p_val = row['p_corrected']
        elif 'p-adj' in row:
            p_val = row['p-adj']
        else:
            p_val = row.get('p', np.nan)
        
        # Format p-value
        if pd.isna(p_val):
            p_str = "p = NA"
        elif p_val < 0.001:
            p_str = "p < .001"
        else:
            p_str = f"p = {p_val:.3f}"
        
        # Cohen's d if available
        d_str = ""
        if 'Cohens_d' in row:
            d_str = f", d = {row['Cohens_d']:.2f}"
        
        sig = " *" if row.get('Significant', False) or row.get('reject', False) else ""
        
        reports.append(f"{g1} vs {g2}: {p_str}{d_str}{sig}")
    
    return reports


# ============================================================================
# PINGOUIN-COMPATIBLE WRAPPER
# ============================================================================

def rm_anova(data, dv=None, within=None, subject=None, detailed=False, aggregate_func='mean'):
    """
    Pingouin-compatible wrapper for RM-ANOVA using rpy2.

    This function provides a drop-in replacement for pingouin.rm_anova() that
    automatically aggregates multiple observations per cell to avoid warnings.

    Parameters
    ----------
    data : DataFrame
        Long-format dataframe
    dv : str
        Dependent variable column name
    within : str or list
        Within-subject factor(s). Can be a single factor (str) or two factors (list)
    subject : str
        Subject identifier column
    detailed : bool
        If True, return detailed output (currently ignored, returns same format)
    aggregate_func : str or callable
        How to aggregate multiple observations per cell ('mean', 'median', etc.)

    Returns
    -------
    DataFrame : ANOVA results table compatible with pingouin format
    """
    if dv is None or within is None or subject is None:
        raise ValueError("dv, within, and subject must be specified")

    # Determine if 1-way or 2-way
    if isinstance(within, list):
        if len(within) == 1:
            # 1-way with list input
            result = run_rm_anova_1way(data, dv, subject=subject, within=within[0],
                                       aggregate_func=aggregate_func)
        elif len(within) == 2:
            # 2-way
            result = run_rm_anova_2way(data, dv, subject=subject, within=within,
                                       aggregate_func=aggregate_func)
        else:
            raise ValueError("within must have 1 or 2 factors")
    else:
        # 1-way with string input
        result = run_rm_anova_1way(data, dv, subject=subject, within=within,
                                   aggregate_func=aggregate_func)

    if result is None:
        # Return empty DataFrame on failure
        return pd.DataFrame()

    # Convert to pingouin-like format
    anova_table = result['anova_table']

    if anova_table is not None:
        # Rename columns to match pingouin format
        pg_format = anova_table.copy()
        pg_format = pg_format.rename(columns={
            'F Value': 'F',
            'Num DF': 'ddof1',
            'Den DF': 'ddof2',
            'Pr > F': 'p-unc',
            'partial_eta_sq': 'np2'
        })

        # Add Source column from index
        pg_format.insert(0, 'Source', pg_format.index)
        pg_format = pg_format.reset_index(drop=True)

        return pg_format
    else:
        return pd.DataFrame()


# ============================================================================
# BATCH ANALYSIS FUNCTIONS
# ============================================================================

def analyze_multiple_metrics_1way(df, metric_cols, subject='Pair', within='Condition',
                                  alpha=0.05, correct_alpha=True):
    """
    Run 1-way RM-ANOVA for multiple metrics with optional Bonferroni correction.
    
    Parameters
    ----------
    df : DataFrame
        Data
    metric_cols : list
        List of metric column names
    subject : str
        Subject identifier
    within : str
        Within-subject factor
    alpha : float
        Significance level
    correct_alpha : bool
        Apply Bonferroni correction across metrics
        
    Returns
    -------
    dict : Results for each metric
    """
    if correct_alpha:
        alpha_corrected = alpha / len(metric_cols)
    else:
        alpha_corrected = alpha
    
    results = {}
    
    for metric in metric_cols:
        print(f"\nAnalyzing {metric}...")
        
        # Average across trials within Pair-Condition
        df_avg = df.groupby([subject, within], as_index=False)[metric].mean()
        
        # Run ANOVA
        anova_res = run_rm_anova_1way(df_avg, dv=metric, subject=subject, within=within)
        
        # Post-hoc if significant
        posthoc_res = None
        if anova_res is not None:
            p_val = anova_res['anova_table'].loc[within, 'Pr > F']
            if p_val < alpha_corrected:
                print(f"  Significant effect found, running post-hoc tests...")
                posthoc_res = bonferroni_pairwise(df_avg, dv=metric, group_col=within, alpha=alpha)
        
        results[metric] = {
            'anova': anova_res,
            'posthoc': posthoc_res,
            'alpha_used': alpha_corrected
        }
    
    return results


def create_results_summary_table(results_dict):
    """
    Create summary table from multiple ANOVA results.
    
    Parameters
    ----------
    results_dict : dict
        Results from analyze_multiple_metrics_1way
        
    Returns
    -------
    DataFrame : Summary table
    """
    rows = []
    
    for metric, res in results_dict.items():
        anova = res['anova']
        if anova is None:
            continue
            
        table = anova['anova_table']
        effect_row = table.iloc[0]  # First effect
        
        rows.append({
            'Metric': metric,
            'F': effect_row['F Value'],
            'df_num': int(effect_row['Num DF']),
            'df_den': int(effect_row['Den DF']),
            'p': effect_row['Pr > F'],
            'partial_eta_sq': effect_row['partial_eta_sq'],
            'Significant': effect_row['Pr > F'] < res['alpha_used']
        })
    
    return pd.DataFrame(rows)

# ---- defaults ----
COND_ORDER = ["L", "M", "H"]
DESIRED_ORDER = ["Vel", "Acc", "Rms"]

NON_METRIC_COLS = {"file","participant","participant_id","condition","window_index",
                   "window_start_s","window_end_s","source","method","_source_file","session"}

# -----------------------
# small helpers
# -----------------------
def split_metric_name(name: str) -> Tuple[str, str]:
    pretty = (name.replace("avg","Average").replace("dist","Amplitude")
                   .replace("mean_abs_","").replace("mean_abs","")
                   .replace("_"," ").title())
    parts = pretty.split()
    return (pretty,"") if len(parts) < 2 else (" ".join(parts[:-1]), parts[-1])

def fmt(beta: float | None, p: float | None) -> Tuple[str,str]:
    # beta now holds SMD (Cohen's d / Hedges-like)
    if beta is None or (isinstance(beta, float) and not np.isfinite(beta)):
        b = "--"
    else:
        b = (f"$d = {beta:.2f}$" if abs(beta) >= 1e-3 else f"$d = {beta:.3g}$")
    if p is None or (isinstance(p,float) and not np.isfinite(p)):
        return b, "--"
    return b, (r"$p < .001$" if p < 0.001 else f"$p = {p:.3f}$")

# def fmt(beta: float | None, p: float | None) -> Tuple[str,str]:
#     b = "--" if beta is None or (isinstance(beta,float) and not np.isfinite(beta)) else f"$\\beta = {beta:.3f}$"
#     if p is None or (isinstance(p,float) and not np.isfinite(p)):
#         return b, "--"
#     return b, (r"$p < .001$" if p < 0.001 else f"$p = {p:.3f}$")

# -----------------------
# rpy2 wrapper: lmer + emmeans
# -----------------------
def _robust_ci_cols(ci_pd: pd.DataFrame) -> Tuple[str|None, str|None]:
    cand = list(ci_pd.columns)
    lower = next((c for c in cand if c.lower().startswith("lower")), None)
    upper = next((c for c in cand if c.lower().startswith("upper")), None)
    if lower and upper:
        return lower, upper
    lower = next((c for c in cand if "lcl" in c.lower()), None)
    upper = next((c for c in cand if "ucl" in c.lower()), None)
    return lower, upper


def run_rpy2_lmer(df: pd.DataFrame, dv: str, adjust: str = "tukey"):
    """
    Fit: dv ~ condition + widx_c + (1 + widx_c || participant)
    Returns:
      pairs_est: dict[(lo,hi)->float]  standardized mean differences (hi - lo)
      pairs_p:   dict[(lo,hi)->float]  p-values from emmeans::pairs(emm)
      means:     dict[cond->float]     emmeans (raw units)
      cis:       dict[cond->(lo,hi)]   95% CI for emmeans (raw units)
    """
    # ---- prep pandas data ----
    need = ["participant", "condition", "window_index", dv]
    d = df[need].dropna().copy()
    d = d.rename(columns={dv: "dv"})
    d["condition"] = pd.Categorical(
        d["condition"].astype(str).str.strip().str.upper(),
        categories=["L","M","H"], ordered=True
    )
    w = pd.to_numeric(d["window_index"], errors="coerce")
    w = (w - np.nanmean(w)) / (np.nanstd(w) if np.nanstd(w) != 0 else 1.0)
    d["widx_c"] = w.fillna(0.0)

    # ---- R bridge ----
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import pandas2ri

    lme4     = importr("lme4")
    lmerTest = importr("lmerTest")
    emmeans  = importr("emmeans")

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["dat"] = ro.conversion.py2rpy(d)

    ro.r('emmeans::emm_options(lmer.df = "satterthwaite", lmerTest.limit = 4000)')
    ro.r('dat$participant <- factor(dat$participant)')
    ro.r('dat$condition   <- factor(dat$condition, levels=c("L","M","H"), ordered=TRUE)')
    ro.r('dat$widx_c      <- as.numeric(dat$widx_c)')

    # ---- fit with random slopes; fallback to intercept-only if singular ----
    ro.r('ctrl <- lme4::lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e6))')
    ro.r('fit_try <- suppressMessages(suppressWarnings('
         '  lmerTest::lmer(dv ~ condition + widx_c + (1 + widx_c || participant), data=dat, control=ctrl)))')
    ro.r('is_sing <- lme4::isSingular(fit_try, tol=1e-6)')
    ro.r('bad_vc  <- any(unlist(lme4::VarCorr(fit_try)) < 1e-10, na.rm=TRUE)')
    ro.r('fit <- if (is_sing || bad_vc) { '
         '  suppressMessages(suppressWarnings('
         '    lmerTest::lmer(dv ~ condition + widx_c + (1 | participant), data=dat, control=ctrl))) '
         '} else { fit_try }')

    # ---- emmeans + pairwise + standardized effect sizes ----
    ro.r('emm <- emmeans::emmeans(fit, ~ condition)')
    ro.r(f'pw  <- pairs(emm, adjust = "{adjust}")')
    ro.r('pw_es <- tryCatch(emmeans::eff_size(pw, sigma = sigma(fit), edf = df.residual(fit)), error=function(e) NULL)')
    ro.r('sig <- as.numeric(sigma(fit))')  # model residual sigma for fallback

    # pull frames
    emm_df_r  = ro.r("as.data.frame(emm)")
    try:
        ci_df_r = ro.r("as.data.frame(confint(emm, level=0.95))")
    except Exception:
        ci_df_r = None
    pwc_df_r  = ro.r("as.data.frame(pw)")
    pwes_df_r = ro.r("if (is.null(pw_es)) data.frame() else as.data.frame(pw_es)")
    sig_r     = float(ro.r("sig")[0])

    with localconverter(ro.default_converter + pandas2ri.converter):
        emm_pd  = ro.conversion.rpy2py(emm_df_r)
        pwc_pd  = ro.conversion.rpy2py(pwc_df_r)
        pwes_pd = ro.conversion.rpy2py(pwes_df_r)
        ci_pd   = ro.conversion.rpy2py(ci_df_r) if ci_df_r is not None else pd.DataFrame()

    # ---- emmeans (raw units) ----
    means = {str(r["condition"]): float(r["emmean"]) for _, r in emm_pd.iterrows()}

    # ---- CIs (raw units) ----
    def _robust_ci_cols(ci_pd: pd.DataFrame):
        cand = list(ci_pd.columns)
        lower = next((c for c in cand if c.lower().startswith("lower")), None)
        upper = next((c for c in cand if c.lower().startswith("upper")), None)
        if not (lower and upper):
            lower = next((c for c in cand if "lcl" in c.lower()), None)
            upper = next((c for c in cand if "ucl" in c.lower()), None)
        return lower, upper

    cis = {}
    if not ci_pd.empty and "condition" in ci_pd.columns:
        lower_col, upper_col = _robust_ci_cols(ci_pd)
        if lower_col and upper_col:
            for _, r in ci_pd.iterrows():
                cis[str(r["condition"])] = (float(r[lower_col]), float(r[upper_col]))
    if not cis:
        se_col = next((c for c in emm_pd.columns if c.lower() in ("se","stderr","std.error")), None)
        if se_col:
            for _, r in emm_pd.iterrows():
                m = float(r["emmean"]); se = float(r[se_col])
                cis[str(r["condition"])] = (m - 1.96*se, m + 1.96*se)
        else:
            for k in means: cis[k] = (float("nan"), float("nan"))

    # ---- pairwise SMDs + p-values ----
    # 1) Build p map from pairs(...)
    pcol = "p.value" if "p.value" in pwc_pd.columns else next((c for c in pwc_pd.columns if c.lower().startswith("p")), None)
    order = {"L":0, "M":1, "H":2}
    pairs_p = {}

    # 2) Robust SMD column detection on pw_es
    smd_col = None
    for candidate in ["effect.size", "SMD", "g", "d", "es", "ES"]:
        if candidate in pwes_pd.columns:
            smd_col = candidate
            break

    # 3) Build maps for SMD (preferred) and raw estimates (fallback)
    smd_map = {}
    if smd_col:
        for _, r in pwes_pd.iterrows():
            contrast = str(r.get("contrast","")).replace("–","-").replace(" - ","-")
            smd_map[contrast] = float(r.get(smd_col, np.nan))

    est_map = {}
    if "estimate" in pwc_pd.columns:
        for _, r in pwc_pd.iterrows():
            contrast = str(r.get("contrast","")).replace("–","-").replace(" - ","-")
            est_map[contrast] = float(r.get("estimate", np.nan))

    # 4) Normalize to (lo,hi) with sign as (hi - lo)
    pairs_est = {}
    for _, r in pwc_pd.iterrows():
        contrast = str(r.get("contrast","")).replace("–","-").replace(" - ","-")
        parts = [p.strip() for p in contrast.split("-")]
        if len(parts) != 2:
            continue
        a, b = parts[0], parts[1]
        if a not in order or b not in order or a == b:
            continue

        # get p
        pv = float(r[pcol]) if (pcol and pd.notnull(r[pcol])) else float("nan")

        # preferred: SMD from eff_size; fallback: estimate/sigma
        est_lr = smd_map.get(contrast, np.nan)
        if not np.isfinite(est_lr) and np.isfinite(sig_r) and sig_r > 0 and contrast in est_map:
            est_lr = est_map[contrast] / sig_r

        lo, hi = (a, b) if order[a] < order[b] else (b, a)
        est_hi_minus_lo = est_lr if (a == hi and b == lo) else -est_lr

        pairs_est[(lo, hi)] = est_hi_minus_lo
        pairs_p[(lo,  hi)]  = pv

    return pairs_est, pairs_p, means, cis


# --------------------------
# plotting helper
# --------------------------
def barplot_ax(ax, means: List[float], sems: List[float], pvals: List[float],
               ylabel: str, metric_name: str,
               colors: List[str] | None = None,
               bar_width: float = 0.80,
               ylim_padding: Tuple[float,float] = (0.4, 0.1)):
    if colors is None:
        colors = ['#4575b4', '#ffffbf', '#d73027']

    import numpy as _np
    x = _np.arange(len(means))

    ax.bar(x, means, yerr=sems, capsize=4, color=colors, width=bar_width, edgecolor="black", linewidth=4)
    
    lowers = [m - (s if not _np.isnan(s) else 0) for m,s in zip(means,sems)]
    uppers = [m + (s if not _np.isnan(s) else 0) for m,s in zip(means,sems)]
    y_min = min(lowers); y_max = max(uppers)
    y_span = y_max - y_min if y_max > y_min else 1.0
    pairs = [(0,1,pvals[0]), (0,2,pvals[1]), (1,2,pvals[2])]
    sig_pairs = [(i,j,p) for (i,j,p) in pairs if (p is not None and not np.isnan(p) and p < 0.05)]
    sig_pairs = sorted(sig_pairs, key=lambda t: (t[1]-t[0]))
    h_step = 0.2 * y_span; line_h = 0.03 * y_span; y0 = y_max + 0.04 * y_span
    for idx, (i,j,p) in enumerate(sig_pairs):
        y = y0 + idx * h_step
        ax.plot([x[i], x[i], x[j], x[j]], [y, y+line_h, y+line_h, y], lw=1.5, color='black', clip_on=False)
        stars = '***' if p < .001 else '**' if p < .01 else '*'
        ax.text((x[i]+x[j])/2, y+0.25*line_h, stars, ha='center', va='bottom', fontsize=13, fontweight='bold', color='black', clip_on=False)
    ax.set_xlim(-0.5, len(means)-0.5); ax.set_xticks([]); ax.set_ylabel("\n".join(textwrap.wrap(ylabel, width=25)), weight='bold', fontsize=12)
    ax.set_ylim(y_min - ylim_padding[0]*y_span, y_max + ylim_padding[1]*y_span + len(sig_pairs)*h_step)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)); ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.spines[['top','right']].set_visible(False)
    for spine in ax.spines.values(): spine.set_linewidth(1.4)
    ax.tick_params(axis='y', width=1.3, labelsize=11)
    for lab in ax.get_yticklabels(): lab.set_fontweight('bold')

# --------------------------
# discovery + loading
# --------------------------
def discover_linear_files(root: Path = Path("data/processed_data")) -> Dict[str, List[Path]]:
    sessions = {}
    root = Path(root)
    for session_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        lm_dir = session_dir / "linear_metrics"
        if lm_dir.exists() and any(lm_dir.glob("*.csv")):
            sessions[session_dir.name] = sorted(lm_dir.glob("*.csv"))
    return sessions

def load_session_csvs(files: List[Path]) -> pd.DataFrame:
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "participant" not in df.columns and "participant_id" in df.columns:
                df = df.rename(columns={"participant_id":"participant"})
            df["_source_file"] = str(f.name)
            parts.append(df)
        except Exception as e:
            print(f"[WARN] failed to load {f}: {e}")
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()

# --------------------------
# table builder + plots (main)
# --------------------------
def build_table_with_emmeans(df: pd.DataFrame, out_tex: str | Path, figs_dir: str | Path):
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import numpy as np # Ensure numpy is imported for np.nan

    out_tex = Path(out_tex)
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    # prepare df
    df = df.copy()
    # Assuming NON_METRIC_COLS, COND_ORDER, run_rpy2_lmer, fmt, split_metric_name,
    # barplot_ax, and DESIRED_ORDER are defined elsewhere in your script.
    df["condition"] = df["condition"].astype(str).str.strip().str.upper()
    df = df[df["condition"].isin(COND_ORDER)].copy()
    if "window_index" not in df.columns:
        df["window_index"] = 0

    for c in df.columns:
        if c not in NON_METRIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    metric_cols = [c for c in df.columns
                   if c not in NON_METRIC_COLS
                   and pd.api.types.is_numeric_dtype(df[c])
                   and df[c].notna().any()]

    # ---- helper: bucket by kinematics (displacement/velocity/acceleration) ----
    def bucket_kind(metric_name: str, metric_type_label: str) -> str:
        mn = metric_name.lower()
        mt = (metric_type_label or "").lower()
        if ("_vel" in mn) or (" velocity" in mt) or (mt == "velocity"):
            return "vel"
        if ("_acc" in mn) or (" acceleration" in mt) or (mt == "acceleration"):
            return "acc"
        return "disp"

    # storage: region -> list of dict rows (we keep kind to split later)
    grouped = defaultdict(list)

    modeled = skipped = 0
    for metric in metric_cols:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        ser = df[metric]
        n_total = ser.shape[0]; n_na = ser.isna().sum()
        if n_na == n_total:
            skipped += 1
            print(f"[skip] {metric}: all NA ({n_na}/{n_total})")
            continue

        sub = df.loc[ser.notna(), ["condition","participant","window_index",metric]]
        conds = sorted(sub["condition"].unique().tolist())
        n_by_cond = sub.groupby("condition")[metric].size().to_dict()
        if not {"L","M","H"}.issubset(set(conds)):
            skipped += 1
            print(f"[skip] {metric}: missing condition(s). have={conds}, counts={n_by_cond}")
            continue

        tmp = sub.rename(columns={metric:"dv"})
        try:
            pairs_est, pairs_p, means, cis = run_rpy2_lmer(tmp, "dv", adjust="tukey")
        except Exception as e:
            skipped += 1
            print(f"[skip] {metric}: model error -> {e}")
            continue

        b_m = pairs_est.get(("L","M"), np.nan); p_m = pairs_p.get(("L","M"), np.nan)
        b_h = pairs_est.get(("L","H"), np.nan); p_h = pairs_p.get(("L","H"), np.nan)
        b_hm= pairs_est.get(("M","H"), np.nan); p_hm= pairs_p.get(("M","H"), np.nan)

        Bm, Pm = fmt(b_m, p_m)
        Bh, Ph = fmt(b_h, p_h)
        Bhm, Phm = fmt(b_hm, p_hm)

        region, metric_type = split_metric_name(metric)
        kind = bucket_kind(metric, metric_type)
        grouped[region].append({
            "metric_type": metric_type,
            "kind": kind,
            "Bm": Bm, "Pm": Pm,
            "Bh": Bh, "Ph": Ph,
            "Bhm": Bhm, "Phm": Phm,
        })
        modeled += 1

        # Plotting logic remains unchanged
        conds = ["L","M","H"]
        mean_vals = [means.get(c, float("nan")) for c in conds]
        sems = []
        for c in conds:
            if c in cis and cis[c] is not None:
                lo, hi = cis[c]
                sems.append((hi - lo) / 3.92 if (not pd.isna(lo) and not pd.isna(hi)) else float("nan"))
            else:
                sems.append(float("nan"))
        pvals_for_plot = [p_m, p_h, p_hm]
        fig, ax = plt.subplots(figsize=(4,5))
        barplot_ax(ax, mean_vals, sems, pvals_for_plot, ylabel=metric.replace("_"," ").title(), metric_name=metric)
        ax.set_title(f"{metric.replace('_',' ').title()}", fontsize=11, weight="bold")
        out_svg = figs_dir / f"{metric}.svg"
        fig.savefig(out_svg, bbox_inches="tight")
        plt.close(fig)

    # ---- New helper to generate and write LaTeX tables for a given mode ----
    def write_tables(mode: str):
        """
        Generates and writes tables.
        mode='complete': Writes all metrics.
        mode='simple': Writes only Mean and RMS metrics.
        """
        if mode == 'simple':
            file_suffix = "_means_rms"
            metrics_to_include = {'mean', 'rms'}
            print(f"\n--- Generating SIMPLE tables (metrics: {', '.join(metrics_to_include)}) ---")
        else:
            file_suffix = "_complete"
            metrics_to_include = None  # A-OK to include all metrics
            print("\n--- Generating COMPLETE tables (all metrics) ---")

        def write_table_for_bucket(bucket: str, bucket_suffix: str):
            lines = [
                r"\begin{tabular}{llcc|cc|cc}",
                r"\toprule",
                r"Region & Metric & $d_{\text{M--L}}$ & $p_{\text{M--L}}$ & $d_{\text{H--L}}$ & $p_{\text{H--L}}$ & $d_{\text{H--M}}$ & $p_{\text{H--M}}$ \\",
                r"\midrule"
            ]
            wrote_any = False
            for region in sorted(grouped.keys()):
                # 1. Filter rows by kinematic bucket (disp, vel, acc)
                rows_in_bucket = [r for r in grouped[region] if r["kind"] == bucket]

                # 2. If simple mode, further filter by metric type (Mean, Rms)
                if metrics_to_include:
                    rows_to_write = [r for r in rows_in_bucket if r['metric_type'].lower() in metrics_to_include]
                else:
                    rows_to_write = rows_in_bucket
                
                if not rows_to_write:
                    continue

                # Sort and write rows to the table
                rows_to_write.sort(key=lambda x: DESIRED_ORDER.index(x["metric_type"]) if x["metric_type"] in DESIRED_ORDER else len(DESIRED_ORDER))
                first = True
                for r in rows_to_write:
                    region_label = f"\\multirow{{{len(rows_to_write)}}}{{*}}{{{region}}}" if first else ""
                    lines.append(f"{region_label} & {r['metric_type']} & {r['Bm']} & {r['Pm']} & {r['Bh']} & {r['Ph']} & {r['Bhm']} & {r['Phm']} \\\\")
                    first = False
                lines.append(r"\midrule")
                wrote_any = True

            lines += [r"\bottomrule", r"\end{tabular}"]

            if wrote_any:
                # Construct the final output path with the correct suffixes
                path = out_tex.with_name(out_tex.stem + file_suffix + bucket_suffix + out_tex.suffix)
                path.write_text("\n".join(lines), encoding="utf-8")
                print(f"  [OK] Wrote {path.name}")
            else:
                print(f"  [NOTE] No rows for bucket='{bucket}' in mode='{mode}'; no table written.")

        # For the given mode, generate tables for each kinematic type
        write_table_for_bucket("disp", "_disp")
        write_table_for_bucket("vel",  "_vel")
        write_table_for_bucket("acc",  "_acc")

    # ---- Generate both sets of tables ----
    write_tables(mode='complete')
    write_tables(mode='simple')

    print(f"\n[DONE] modeled={modeled}, skipped={skipped}")
    return modeled, skipped


def pretty_metric(name: str) -> str:
    """Format metric names for display."""
    return name.replace("_", " ").title()


def run_stats_by_column(
    df: pd.DataFrame,
    metrics,
    *,
    adjust: str = "tukey",
    min_per_condition: int = 1,
    verbose: bool = True
):
    """
    For each data 'column' and each metric, fit:
        dv ~ condition + widx_c + (1 + widx_c || participant)
    with automatic fallback to (1|participant), emmeans, pairwise tests,
    and standardized effect sizes (via run_rpy2_lmer).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ["participant","condition","window_index","column", <metrics...>]
    metrics : Iterable[str]
        Column names to use as DVs.
    adjust : str
        p-value adjustment method passed to emmeans::pairs (default "tukey").
    min_per_condition : int
        Minimum rows per condition to attempt a model (after NA drop).
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict
        Nested dict: results[column][metric] = (pairs_est, pairs_p, means, cis)
          - pairs_est : dict[(lo,hi)->float] standardized mean differences (hi - lo)
          - pairs_p   : dict[(lo,hi)->float] p-values
          - means     : dict[cond->float]     emmeans (raw units)
          - cis       : dict[cond->(lo,hi)]   95% CI for emmeans (raw units)
    """
    results = defaultdict(dict)

    # sanity: ensure required columns exist
    required = {"participant", "condition", "window_index", "column"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    # iterate deterministically by column label
    for col_name in sorted(pd.Series(df["column"]).dropna().unique().tolist()):
        if verbose:
            print(f"\nAnalyzing: {col_name}")
        dsub = df[df["column"] == col_name].copy()

        # must have at least 2 conditions in this slice (after cleaning)
        conds_raw = pd.Series(dsub["condition"]).dropna().astype(str)
        if conds_raw.empty:
            if verbose: print("  Skipping (no condition values)")
            continue

        # iterate metrics
        for metric in metrics:
            if metric not in dsub.columns:
                if verbose: print(f"  – {metric}: not in dataframe; skipping")
                continue

            # build minimal frame and drop NAs
            need = ["participant", "condition", "window_index", metric]
            tmp = dsub[need].dropna().copy()
            if tmp.empty:
                if verbose: print(f"  – {metric}: empty after NA drop; skipping")
                continue

            # normalize condition coding early (mirrors run_rpy2_lmer)
            tmp["condition"] = (
                tmp["condition"].astype(str).str.strip().str.upper()
            )

            # keep only conditions that have at least min_per_condition rows
            counts = tmp.groupby("condition", dropna=True).size()
            ok_levels = counts[counts >= max(1, int(min_per_condition))].index.tolist()
            tmp = tmp[tmp["condition"].isin(ok_levels)]

            uniq_conds = sorted(tmp["condition"].unique().tolist())
            if len(uniq_conds) < 2:
                if verbose:
                    n_conds = len(uniq_conds)
                    print(f"  Skipping {metric} (only {n_conds} usable condition(s) after filtering)")
                continue

            try:
                # delegate all modeling/details to run_rpy2_lmer
                pairs_est, pairs_p, means, cis = run_rpy2_lmer(
                    tmp.rename(columns={metric: "dv"}),
                    dv="dv",
                    adjust=adjust
                )
                results[col_name][metric] = (pairs_est, pairs_p, means, cis)
                if verbose:
                    # quick, informative summary
                    have_es = ", ".join([f"{a}>{b}" for (a,b) in sorted(pairs_est)])
                    print(f"  ✓ {metric} — contrasts: [{have_es}]")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {metric}: {type(e).__name__}: {e}")

    return results
