import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import warnings

## get df with metrics that cant be compared to unperturbed environment
def get_metrics_robustness_not_compared_to_unperturbed(metrics_models, model_names=None, print_as_latex=False, **kwargs):
    """
    Generate a table with robustness metrics that cannot be directly compared to the unperturbed environment.
    Metrics include number of actions changed per 1000 steps and similarity score per changed action.
    
    Args:
    - metrics_models: list of Metrics objects for different models
    - model_names: optional list of names for the models; if None, names from Metrics objects are used
    - print_as_latex: if True, prints the table in LaTeX format
    - kwargs: additional arguments for the df_to_latex function

    Returns:
    - table_metrics: DataFrame with the calculated robustness metrics
    """
    metrics_list = []
    for i in range(len(metrics_models)):
        metrics_model = metrics_models[i]
        # Calculate mean values for relevant metrics
        table_metrics = metrics_model.metrics_robustness[["similarity_score", "n_actions_changed", "n_steps"]].mean()
        # Compute actions changed per 1000 steps
        table_metrics["n_act_change_per_1000"] = table_metrics["n_actions_changed"] / table_metrics["n_steps"] * 1000
        # Compute similarity score per action changed
        if table_metrics["n_actions_changed"] > 0:
            table_metrics["similarity_score_per_act"] = table_metrics["similarity_score"] / table_metrics["n_actions_changed"]
        else:
            table_metrics["similarity_score_per_act"] = np.nan
            if model_names is None:
                name = metrics_model.model_name
            else:
                name = model_names[i]
            warnings.warn(f"No actions changed by model {name} so no similarity score is available")
        metrics_list.append(table_metrics[["n_act_change_per_1000", "similarity_score_per_act"]])

    # Set model names if not provided
    if model_names is None:
        model_names = [metrics.model_name for metrics in metrics_models]
    table_metrics = pd.DataFrame(metrics_list, index=model_names).T

    # Optionally print as LaTeX
    if print_as_latex:
        df_to_latex(table_metrics, **kwargs)
    
    return table_metrics

def plot_metrics_robustness_compared_to_unperturbed(metrics_models, metrics_unperturbed, model_names=None, filename=""):
    """
    Plot robustness metrics of perturbed models as a percentage of the unperturbed model.
    Metrics include total reward and number of steps.
    
    Args:
    - metrics_models: list of Metrics objects for different models
    - metrics_unperturbed: Metrics object for the unperturbed model
    - model_names: optional list of names for the models; if None, names from Metrics
        objects are used
    - filename: if provided, saves the plot to this file
    """
    # Get unperturbed metrics
    metrics_unperturbed = metrics_unperturbed.metrics_robustness[["total_reward", "n_steps"]].copy()
    comparison_mean_metrics = []
    for metrics_model in metrics_models:
        # Get perturbed metrics
        metrics_perturbed = metrics_model.metrics_robustness[["total_reward", "n_steps"]].copy()
        # Calculate percentage relative to unperturbed
        metrics_perturbed = (metrics_perturbed / metrics_unperturbed) * 100
        comparison_mean_metrics.append(metrics_perturbed.mean().values)

    # Set model names if not provided
    if model_names is None:
        model_names = [metrics.model_name for metrics in metrics_models]
    comparison_mean_metrics = pd.DataFrame(comparison_mean_metrics, columns=["Total reward", "Number of steps"], index=model_names).T

    # Plot bar chart
    fig, ax = plt.subplots(1, 1, figsize=(9,4))
    comparison_mean_metrics.plot.bar(rot=0, ax=ax)
    ax.set_ylabel("Performance on metric as a percentage \n of unperturbed performance (%)")
    fig.suptitle("Performance of AI system when including perturbations compared\n to unperturbed situation")
    ax.legend(loc=(1.02,0.4))
    fig.tight_layout()
    if filename != "":
        fig.savefig(filename)
    plt.close(fig)

def get_metrics_resilience(metrics_models, model_names=None, print_as_latex=False, **kwargs):
    """
    Generate tables with resilience metrics based on both reward and observation similarity.
    Returns two DataFrames: one for reward-based metrics and one for observation similarity.
    
    Args:
    - metrics_models: list of Metrics objects for different models
    - model_names: optional list of names for the models; if None, names from Metrics objects are used
    - print_as_latex: if True, prints the tables in LaTeX format
    - kwargs: additional arguments for the df_to_latex function

    Returns:
    - table_metrics_reward: DataFrame with resilience metrics based on reward
    - table_metrics_obs_sim: DataFrame with resilience metrics based on observation similarity
    """
    if model_names is None:
        model_names = [metrics.model_name for metrics in metrics_models]

    # Concatenate reward-based resilience metrics
    table_metrics_reward = pd.concat([metrics_model.metrics_resilience for metrics_model in metrics_models]).T
    table_metrics_reward.columns = model_names

    # Concatenate observation similarity-based resilience metrics
    table_metrics_obs_sim = pd.concat([metrics_model.metrics_resilience_obs_sim for metrics_model in metrics_models]).T
    table_metrics_obs_sim.columns = model_names

    # Optionally print as LaTeX
    if print_as_latex:
        print("Metrics resilience based on reward:")
        df_to_latex(table_metrics_reward, **kwargs)
        print("Metrics resilience based on observation similarity:")
        df_to_latex(table_metrics_obs_sim, **kwargs)

    return table_metrics_reward, table_metrics_obs_sim

def plot_reward_curve_comparison(metrics, ep, model_name="", filename="", plot_range=(0, -1)):
    """
    Plot a comparison of reward per step between perturbed and unperturbed environments for a given episode.
    
    Args:
    - metrics: Metrics object containing reward data
    - ep: episode index to plot
    - model_name: name of the perturbed model; if empty, uses metrics.model_name
    - filename: if provided, saves the plot to this file
    - plot_range: tuple indicating the range of steps to plot (start, end); defaults to full range
    """
    y_label = "Reward in step"
    title = "Comparison of Reward in each step"
    if model_name == "":
        model_name = metrics.model_name
    plot_curve(metrics.rewards_perturbed[ep], metrics.rewards_unperturbed[ep], y_label, title, model_name, filename=filename, plot_range=plot_range)

def plot_cos_similarity_curve_comparison(metrics, ep, model_name="", filename="", plot_range=(0, -1)):
    """
    Plot cosine similarity of observed state to unperturbed state for a given episode.
    
    Args:
    - metrics: Metrics object containing cosine similarity data
    - ep: episode index to plot
    - model_name: name of the perturbed model; if empty, uses metrics.model_name
    - filename: if provided, saves the plot to this file
    - plot_range: tuple indicating the range of steps to plot (start, end); defaults to full range
    """
    y_label = "Cosine similarity to unperturbed state"
    title = "Cosine similarity of observed state to unperturbed situation"
    if model_name == "":
        model_name = metrics.model_name
    plot_curve(metrics.cos_similarity_all[ep], np.ones_like(metrics.cos_similarity_all[ep]), y_label, title, model_name, filename=filename, plot_range=plot_range)

def plot_curve(data_perturbed, data_unperturbed, y_label, title, model_name, filename="", plot_range=(0, -1)):
    """
    Plot a curve comparing perturbed and unperturbed data over steps.
    
    Args:
    - data_perturbed: array of data points for the perturbed environment
    - data_unperturbed: array of data points for the unperturbed environment
    - y_label: label for the y-axis
    - title: title of the plot
    - model_name: name of the perturbed model
    - filename: if provided, saves the plot to this file
    - plot_range: tuple indicating the range of steps to plot (start, end); defaults to full range
    """
    data_to_show_unperturbed = data_unperturbed[plot_range[0]:plot_range[1]]
    data_to_show_perturbed = data_perturbed[plot_range[0]:plot_range[1]]
    fig, ax = plt.subplots(1, 1, figsize=(8,3.5))
    ax.plot(list(range(len(data_to_show_unperturbed))), data_to_show_unperturbed)
    ax.plot(list(range(len(data_to_show_perturbed))), data_to_show_perturbed)
    ax.legend(["Unperturbed", model_name], loc=(1.01, 0.45))
    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    fig.suptitle(title)
    fig.tight_layout()
    if filename != "":
        fig.savefig(filename)
    plt.close(fig)

def df_to_latex(df, incl_index=True, n_dec=3, multirow=False):
    """
    Print a pandas DataFrame as LaTeX table.
    
    Args:
    - df: pandas DataFrame to convert
    - incl_index: if True, includes the DataFrame index in the LaTeX output
    - n_dec: number of decimal places for floating point numbers
    - multirow: if True, uses multirow formatting for the index
    """
    print(df.to_latex(index=incl_index, float_format=lambda x: f"{x:.{n_dec}f}", escape=False, multirow=multirow))