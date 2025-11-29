import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from math import sqrt
from scipy.stats import chi2_contingency, norm

def z_test_and_effect(df, num_features, target="cardio"):
    """
    Perform two-sample Z-test and compute Cohen's d effect size for numerical features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the features and target.
    num_features : list
        List of numerical feature names to test.
    target : str
        Binary target column (default 'cardio').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Feature, Z-score, p-value, Cohen_d, Effect_strength].
    """
    results = []

    for col in num_features:
        g0 = df[df[target] == 0][col]
        g1 = df[df[target] == 1][col]

        s1, s2 = len(g0), len(g1)
        mean0, mean1 = g0.mean(), g1.mean()
        std0, std1 = g0.std(), g1.std()

        # Standard error and z-score
        se = sqrt(std0**2 / s1 + std1**2 / s2)
        z = (mean0 - mean1) / se
        p_val = 2 * (1 - norm.cdf(abs(z)))

        # Cohen's d
        pooled_std = sqrt(((s1 - 1) * std0**2 + (s2 - 1) * std1**2) / (s1 + s2 - 2))
        cohen_d = (mean1 - mean0) / pooled_std

        # Effect strength interpretation
        if abs(cohen_d) < 0.2:
            strength = "Very small"
        elif abs(cohen_d) < 0.5:
            strength = "Small"
        elif abs(cohen_d) < 0.8:
            strength = "Medium"
        else:
            strength = "Large"

        results.append({
            "Feature": col,
            "Z-score": z,
            "p-value": p_val,
            "Cohen_d": cohen_d,
            "Effect_strength": strength
        })

    return pd.DataFrame(results).sort_values("Cohen_d", ascending=False).reset_index(drop=True)


def chi_square_and_cramers_v(df, cat_features, target="cardio"):
    """
    Perform Chi-square test and compute Cramer's V for categorical features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the features and target.
    cat_features : list
        List of categorical feature names to test.
    target : str
        Binary target column (default 'cardio').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Feature, Chi2, p-value, Cramers_V].
    """
    results = []

    for col in cat_features:
        contingency = pd.crosstab(df[col], df[target])
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        n = contingency.sum().sum()
        k = min(contingency.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1)))

        results.append({
            "Feature": col,
            "Chi2": chi2,
            "p-value": p_val,
            "Cramers_V": cramers_v
        })

    return pd.DataFrame(results).sort_values("Cramers_V", ascending=False).reset_index(drop=True)

def plot_cohens_d(effect_df):
    """
    Plot Cohen's d effect sizes for numerical features.

    Parameters
    ----------
    effect_df : pandas.DataFrame
        Output from z_test_and_effect() containing ['Feature', 'Cohen_d'].

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Horizontal bar chart showing Cohen's d for each numerical feature.
    """
    fig = px.bar(
        effect_df.sort_values("Cohen_d", ascending=True),
        x="Cohen_d",
        y="Feature",
        orientation="h",
        text="Cohen_d",
        color="Effect_strength",
        color_discrete_map={
            "Very small": "#91c9f7",
            "Small": "#4fa3d1",
            "Medium": "#2b6ca3",
            "Large": "#0a417a"
        },
        title="Effect Size (Cohen's d) for Numerical Features"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Cohen's d",
        yaxis_title="Features",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_cramers_v(cat_df):
    """
    Plot Cramer's V association strength for categorical features.

    Parameters
    ----------
    cat_df : pandas.DataFrame
        Output from chi_square_and_cramers_v() containing ['Feature', 'Cramers_V'].

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Horizontal bar chart showing Cramer's V for each categorical feature.
    """
    fig = px.bar(
        cat_df.sort_values("Cramers_V", ascending=True),
        x="Cramers_V",
        y="Feature",
        orientation="h",
        text="Cramers_V",
        color_discrete_sequence=["#2a7f62"],
        title="Association Strength (Cramer's V) for Categorical Features"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Cramer's V",
        yaxis_title="Features",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def plot_feature_distribution(df, columns):
    """Return a list of Seaborn histograms for selected numeric features."""
    figs = []
    for col in columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], kde=True, ax=ax, color="#4fa3d1")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        figs.append(fig)
    return figs

def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numeric columns."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True,
                cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


def plot_target_relationships(df, target, features):
    """Return boxplots showing relationship between target and given features."""
    figs = []
    for col in features:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=target, y=col, data=df, palette="Set2", ax=ax)
        ax.set_title(f"{col} vs {target}")
        figs.append(fig)
    return figs


def plot_categorical_counts(df, cat_features):
    """Plot countplots for categorical features."""
    figs = []
    for col in cat_features:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=col, hue="cardio", data=df, palette="coolwarm", ax=ax)
        ax.set_title(f"{col} by Cardio")
        figs.append(fig)
    return figs

def plot_age_group_distribution(df):
    """Plot countplot of age groups vs cardio."""
    bins = [28, 39, 49, 59, 69]
    labels = ['30-39', '40-49', '50-59', '60-69']
    df["age_group"] = pd.cut(df["age"] / 365, bins=bins, labels=labels)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x="age_group", hue="cardio", data=df, palette="coolwarm", ax=ax)
    ax.set_title("Cardiovascular Disease Cases by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Count")
    return fig
