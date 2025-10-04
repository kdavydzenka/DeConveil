import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_volcano(plot_data, lfc_cut=1.0, pval_cut=0.05, xlim=None, ylim=None):
    plot_data['gene_group'] = plot_data['gene_group'].astype('category')
    
    # Define gene group colors
    gene_group_colors = {
        "DIGs": "#8F3931FF",
        "DSGs": "#FFB977",
        "DCGs": "#FFC300"
    }

    # Create a FacetGrid for faceted plots
    g = sns.FacetGrid(
        plot_data, 
        col="method", 
        margin_titles=True, 
        hue="gene_group", 
        palette=gene_group_colors, 
        sharey=False, 
        sharex=True
    )

    
    # Add points for "DIGs" 
    g.map_dataframe(
        sns.scatterplot, 
        x="log2FC", 
        y="-log10(padj)", 
        alpha=0.2, 
        size=0.5, 
        legend=False, 
        data=plot_data[plot_data['gene_group'] == "DIGs"]
    )

    # Add points for "DSGs" and "DCGs
    g.map_dataframe(
        sns.scatterplot, 
        x="log2FC", 
        y="-log10(padj)", 
        alpha=0.8, 
        s=3.0, 
        legend=False, 
        data=plot_data[plot_data['gene_group'] == "DSGs"]
    )

    g.map_dataframe(
        sns.scatterplot, 
        x="log2FC", 
        y="-log10(padj)", 
        alpha=1.0, 
        s=3.0, 
        legend=False, 
        data=plot_data[plot_data['gene_group'] == "DCGs"],
        zorder=5              # force to front
    )
    
    # Threshold lines
    for ax in g.axes.flat:
        ax.axvline(x=-lfc_cut, color="gray", linestyle="dashed")
        ax.axvline(x=lfc_cut, color="gray", linestyle="dashed")
        ax.axhline(y=-np.log10(pval_cut), color="gray", linestyle="dashed")
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
    
    # Labels and legend
    g.set_axis_labels("Log2 FC", "-Log10 P-value")
    g.add_legend(title="Gene category")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.tight_layout()
    
    # Axis formatting
    for ax in g.axes.flat:
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel("Log2 FC", fontsize=14)
        ax.set_ylabel("-Log10 P-value", fontsize=14)
    
    # Save or display the plot
    plt.show()


def plot_cnv_hist(cnv_mean, binwidth=0.2, title="CNV Mean Distribution"):
    """
    Plots a histogram of the CNV mean distribution.

    Parameters:
        cnv_mean (pd.Series or list): The CNV mean values to plot.
        binwidth (float): The bin width for the histogram.
        title (str): The title of the plot.
    """
    # Convert to a DataFrame if it's not already
    if isinstance(cnv_mean, list):
        cnv_mean = pd.DataFrame({'cnv_mean': cnv_mean})
    elif isinstance(cnv_mean, pd.Series):
        cnv_mean = cnv_mean.to_frame(name='cnv_mean')

    # Create the histogram plot
    plt.figure(figsize=(5, 5))
    sns.histplot(
        cnv_mean['cnv_mean'],
        bins=int((cnv_mean['cnv_mean'].max() - cnv_mean['cnv_mean'].min()) / binwidth),
        kde=False,
        color="#F39B7F",
        edgecolor="black",
        alpha=0.7
    )

    # Add labels and titles
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("CN state", fontsize=14, labelpad=8)
    plt.ylabel("Frequency", fontsize=14, labelpad=8)

    # Customize the appearance of axes
    plt.xticks(fontsize=12, color="black", rotation=45, ha="right")
    plt.yticks(fontsize=12, color="black")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)

    # Add a grid
    plt.grid(visible=False)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_stacked_bar(combined_data):
    """
    Creates a stacked bar plot of gene counts by CNV group for each tumor type.
    
    Parameters:
    - combined_data: DataFrame containing the data to plot.
    """
    # Define CNV colors inside the function
    cnv_colors = {
        "loss": "dodgerblue",
        "neutral": "gray",
        "gain": "yellowgreen",
         "amplification": "coral"
    }
    
    tumor_types = combined_data['tumor_type'].unique()
    
    # Create subplots for each tumor type
    fig, axes = plt.subplots(1, len(tumor_types), figsize=(5, 5), sharey=True)
    
    # If there's only one tumor type, axes will not be an array, so we convert it into a list
    if len(tumor_types) == 1:
        axes = [axes]
    
    for idx, tumor_type in enumerate(tumor_types):
        ax = axes[idx]
        tumor_data = combined_data[combined_data['tumor_type'] == tumor_type]
        
        # Create a table of counts for CNV group vs gene group
        counts = pd.crosstab(tumor_data['gene_group'], tumor_data['cnv_group'])
        
        # Plot stacked bars
        counts.plot(kind='bar', stacked=True, ax=ax, color=[cnv_colors[group] for group in counts.columns], width=0.6)

        ax.set_title(tumor_type, fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("Gene Counts", fontsize=16)
        
        # Customize axis labels and tick marks
        ax.tick_params(axis='x', labelsize=16, labelcolor="black")
        ax.tick_params(axis='y', labelsize=16, labelcolor="black")
    
    # Overall settings for layout and labels
    plt.xticks(fontsize=12, color="black", rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    

def plot_percentage_bar(barplot_data):
    """
    Creates a bar plot showing the percentage of genes for each gene group across tumor types.
    
    Parameters:
    - barplot_data: DataFrame containing 'gene_group', 'percentage', and 'Count' columns.
    """
    # Define the gene group colors inside the function
    gene_group_colors = {
        "DIGs": "#8F3931FF",
        "DSGs": "#FFB977",
        "DCGs": "#FFC300"
    }

    tumor_types = barplot_data['tumor_type'].unique()
    
    plt.figure(figsize=(5, 5))
    sns.set(style="whitegrid")

    # Create subplots for each tumor type
    fig, axes = plt.subplots(1, len(tumor_types), figsize=(5, 5), sharey=True)
    
    # If only one tumor type, ensure axes is a list
    if len(tumor_types) == 1:
        axes = [axes]
    
    for idx, tumor_type in enumerate(tumor_types):
        ax = axes[idx]
        tumor_data = barplot_data[barplot_data['tumor_type'] == tumor_type]
        
        # Plot the percentage bar plot
        sns.barplot(data=tumor_data, x="gene_group", y="percentage", hue="gene_group",
                    palette=gene_group_colors, ax=ax, width=0.6)

        # Add counts and percentages as labels
        for p in ax.patches:
            height = p.get_height()
            gene_group = p.get_x() + p.get_width() / 2  # Get the x position of the patch (bar)

            # Find the gene_group in the data based on its position
            group_name = tumor_data.iloc[int(gene_group)]['gene_group']
            count = tumor_data.loc[tumor_data['gene_group'] == group_name, 'Count'].values[0]
            percentage = tumor_data.loc[tumor_data['gene_group'] == group_name, 'percentage'].values[0]

            # Position the labels slightly above the bars
            ax.text(p.get_x() + p.get_width() / 2, height + 0.5, f'{count} ({round(percentage, 1)}%)', 
                    ha='center', va='bottom', fontsize=12, color="black")
        
        ax.set_title(tumor_type, fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("Percentage of Genes", fontsize=16)

        # Customize axis labels and tick marks
        ax.tick_params(axis='x', labelsize=16, labelcolor="black", rotation=45)
        ax.tick_params(axis='y', labelsize=16, labelcolor="black")

        # Explicitly set the x-tick labels with proper rotation and alignment
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')  # This ensures proper alignment for x-ticks
            tick.set_rotation(45)

    # Overall settings for layout and labels
    plt.tight_layout()
    plt.show()


def plot_pca_clusters(pca_coords, labels, explained_var, title="PCA Clustering"):
    """
    Scatterplot of first 2 PCs with cluster colors, showing variance explained.

    Parameters
    ----------
    pca_coords : DataFrame
        PCA coordinates (samples × PCs), from pca_cluster_cn().
    labels : pd.Series
        Cluster assignments (index must match pca_coords).
    explained_var : array-like
        Explained variance ratio for each PC.
    title : str
        Plot title.
    """
    df_plot = pca_coords.copy()
    df_plot["cluster"] = labels.astype(str)

    plt.figure(figsize=(7,6))
    sns.scatterplot(
        x="PC1", y="PC2", hue="cluster",
        data=df_plot, palette="Set2", s=70, alpha=0.9, edgecolor="k"
    )

    # Format axis labels with variance %
    pc1_var = explained_var[0] * 100
    pc2_var = explained_var[1] * 100
    plt.xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    plt.ylabel(f"PC2 ({pc2_var:.1f}% variance)")

    plt.title(title, fontsize=14)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_consensus_matrix(consensus_matrix, labels, title="Consensus Matrix"):
    """
    Heatmap of consensus matrix with samples ordered by cluster.
    
    Parameters
    ----------
    consensus_matrix : pd.DataFrame
        Sample × Sample consensus values.
    labels : pd.Series
        Final cluster assignments (same sample index).
    """
    # Order samples by cluster
    ordered_samples = labels.sort_values().index
    mat = consensus_matrix.loc[ordered_samples, ordered_samples]

    plt.figure(figsize=(7,6))
    sns.heatmap(mat, cmap="viridis", square=True, cbar_kws={"label": "Consensus"})
    plt.title(title, fontsize=14)
    plt.xlabel("Samples")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.show()