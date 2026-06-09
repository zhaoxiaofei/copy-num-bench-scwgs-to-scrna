#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def auto_classify_cells(
    segcopy_path, 
    output_path, 
    exclude_sex_chromosomes=True, 
    method="kmeans_diploid_percent", 
    manual_threshold=None
):
    """
    Automatically classify cells as tumor or normal based on CNV patterns
    
    Args:
        segcopy_path: Path to Ginkgo SegCopy file
        output_path: Path to save the output annotation file
        exclude_sex_chromosomes: Whether to exclude X/Y chromosomes from calculation
        method: Classification method to use:
            - "kmeans_diploid_percent": K-means clustering on diploid percentage (RECOMMENDED)
            - "kmeans_deviation": K-means clustering on CNV deviation score
            - "threshold_diploid_percent": Manual threshold on diploid percentage
            - "threshold_deviation": Manual threshold on CNV deviation score
        manual_threshold: Manual threshold value when using threshold methods
    """
    print("Reading SegCopy file...")
    
    # Read the SegCopy file
    df = pd.read_csv(segcopy_path, sep='\t')
    
    # Extract clean cell names from Ginkgo's long column names
    cell_names = []
    for col in df.columns[3:]:
        if '..' in col:
            cell_name = col.split('..')[-1]
            cell_names.append(cell_name)
        else:
            cell_names.append(col)
    
    # Rename columns to clean cell names
    df.columns = ['CHR', 'START', 'END'] + cell_names
    
    # Exclude sex chromosomes if requested
    if exclude_sex_chromosomes:
        autosomes = [f'chr{i}' for i in range(1, 23)] + [str(i) for i in range(1, 23)]
        df = df[df['CHR'].isin(autosomes)]
        print(f"Excluded sex chromosomes. Using {len(df)} autosomal segments for analysis.")
    
    # Calculate segment lengths for weighted calculations
    df['LENGTH'] = df['END'] - df['START']
    total_genome_length = df['LENGTH'].sum()
    
    # Calculate metrics for each cell
    print("Calculating CNV metrics...")
    cnv_data = df[cell_names].values
    segment_lengths = df['LENGTH'].values.reshape(-1, 1)
    
    # Metric 1: CNV Deviation Score (weighted by segment length)
    # Sum of absolute differences from diploid (2), weighted by segment length
    deviation_scores = np.sum(np.abs(cnv_data - 2) * segment_lengths, axis=0) / total_genome_length * 1000
    
    # Metric 2: Diploid Percentage (weighted by segment length)
    # Percentage of genome with copy number exactly equal to 2
    # We use a small tolerance (0.1) to account for floating point errors
    diploid_mask = np.abs(cnv_data - 2) < 0.1
    diploid_percentages = np.sum(diploid_mask * segment_lengths, axis=0) / total_genome_length * 100
    
    # Create results DataFrame
    results = pd.DataFrame({
        'cell': cell_names,
        'deviation_score': deviation_scores,
        'diploid_percent': diploid_percentages
    })
    
    # Perform classification
    print(f"Performing classification using method: {method}...")
    results['label'] = 'unknown'
    
    if method == "kmeans_diploid_percent":
        # K-means clustering on diploid percentage (RECOMMENDED)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(diploid_percentages.reshape(-1, 1))
        
        # Normal cells have higher diploid percentage
        cluster_means = [np.mean(diploid_percentages[labels == i]) for i in range(2)]
        normal_cluster = np.argmax(cluster_means)
        tumor_cluster = 1 - normal_cluster
        
        results['label'] = np.where(labels == tumor_cluster, 'tumor', 'normal')
        
        # Calculate threshold between clusters
        auto_threshold = np.mean(kmeans.cluster_centers_)
        print(f"K-means clustering completed successfully")
        print(f"Normal cluster mean diploid percentage: {cluster_means[normal_cluster]:.1f}%")
        print(f"Tumor cluster mean diploid percentage: {cluster_means[tumor_cluster]:.1f}%")
        print(f"Automatic classification threshold: {auto_threshold:.1f}%")
        
    elif method == "kmeans_deviation":
        # K-means clustering on deviation score
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(deviation_scores.reshape(-1, 1))
        
        # Tumor cells have higher deviation scores
        cluster_means = [np.mean(deviation_scores[labels == i]) for i in range(2)]
        tumor_cluster = np.argmax(cluster_means)
        normal_cluster = 1 - tumor_cluster
        
        results['label'] = np.where(labels == tumor_cluster, 'tumor', 'normal')
        
        auto_threshold = np.mean(kmeans.cluster_centers_)
        print(f"K-means clustering completed successfully")
        print(f"Normal cluster mean deviation score: {cluster_means[normal_cluster]:.2f}")
        print(f"Tumor cluster mean deviation score: {cluster_means[tumor_cluster]:.2f}")
        print(f"Automatic classification threshold: {auto_threshold:.2f}")
        
    elif method == "threshold_diploid_percent":
        # Manual threshold on diploid percentage
        if manual_threshold is None:
            # Default threshold: 80% diploid
            auto_threshold = 80.0
            print(f"Using default diploid percentage threshold: {auto_threshold:.1f}%")
        else:
            auto_threshold = manual_threshold
            print(f"Using manual diploid percentage threshold: {auto_threshold:.1f}%")
        
        # Cells with < threshold% diploid are classified as tumor
        results['label'] = np.where(diploid_percentages < auto_threshold, 'tumor', 'normal')
        
    elif method == "threshold_deviation":
        # Manual threshold on deviation score
        if manual_threshold is None:
            # Default threshold: median + 2 standard deviations
            median_dev = np.median(deviation_scores)
            std_dev = np.std(deviation_scores)
            auto_threshold = median_dev + 2 * std_dev
            print(f"Using automatic deviation score threshold: {auto_threshold:.2f} (median + 2σ)")
        else:
            auto_threshold = manual_threshold
            print(f"Using manual deviation score threshold: {auto_threshold:.2f}")
        
        # Cells with > threshold deviation are classified as tumor
        results['label'] = np.where(deviation_scores > auto_threshold, 'tumor', 'normal')
    
    else:
        print(f"Error: Unknown method '{method}'")
        print("Available methods: kmeans_diploid_percent, kmeans_deviation, threshold_diploid_percent, threshold_deviation")
        sys.exit(1)
    
    # Print classification statistics
    tumor_count = sum(results['label'] == 'tumor')
    normal_count = sum(results['label'] == 'normal')
    total_count = len(results)
    
    print("\n=== Classification Results ===")
    print(f"Total cells analyzed: {total_count}")
    print(f"Tumor cells: {tumor_count} ({tumor_count/total_count*100:.1f}%)")
    print(f"Normal cells: {normal_count} ({normal_count/total_count*100:.1f}%)")
    
    # Save the annotation file
    results[['cell', 'label']].to_csv(output_path, sep='\t', header=False, index=False)
    print(f"\nAnnotation file saved to: {output_path}")
    
    # Generate diagnostic plots
    print("Generating diagnostic plots...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Diploid Percentage Distribution
    ax1.hist(diploid_percentages, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    if "diploid_percent" in method:
        ax1.axvline(auto_threshold, color='red', linestyle='--', linewidth=2, label='Classification Threshold')
    ax1.set_xlabel('Diploid Percentage (%)')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Distribution of Diploid Genome Percentage')
    ax1.legend()
    
    # Plot 2: Deviation Score Distribution
    ax2.hist(deviation_scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    if "deviation" in method:
        ax2.axvline(auto_threshold, color='red', linestyle='--', linewidth=2, label='Classification Threshold')
    ax2.set_xlabel('CNV Deviation Score')
    ax2.set_ylabel('Number of Cells')
    ax2.set_title('Distribution of CNV Deviation Scores')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = output_path + '.classification_plots.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Diagnostic plots saved to: {plot_path}")
    
    # Save detailed metrics for manual inspection
    metrics_path = output_path + '.cell_metrics.tsv'
    results.to_csv(metrics_path, sep='\t', index=False)
    print(f"Detailed cell metrics saved to: {metrics_path}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print(f"  python3 {sys.argv[0]} <SegCopy_file> <output_annotation_file> [method] [threshold]")
        print("\nMethods (RECOMMENDED: kmeans_diploid_percent):")
        print("  kmeans_diploid_percent    K-means clustering on diploid percentage (default)")
        print("  kmeans_deviation          K-means clustering on CNV deviation score")
        print("  threshold_diploid_percent Manual threshold on diploid percentage")
        print("  threshold_deviation       Manual threshold on CNV deviation score")
        print("\nExamples:")
        print("  # Recommended: Use K-means on diploid percentage")
        print(f"  python3 {sys.argv[0]} SegCopy cell_annotation.txt")
        print("\n  # Manual threshold: Cells with <75% diploid are tumor")
        print(f"  python3 {sys.argv[0]} SegCopy cell_annotation.txt threshold_diploid_percent 75")
        sys.exit(1)
    
    segcopy_file = sys.argv[1]
    output_file = sys.argv[2]
    
    method = "kmeans_diploid_percent"
    manual_threshold = None
    
    if len(sys.argv) >= 4:
        method = sys.argv[3]
    
    if len(sys.argv) >= 5:
        try:
            manual_threshold = float(sys.argv[4])
        except ValueError:
            print("Error: Threshold must be a number")
            sys.exit(1)
    
    auto_classify_cells(
        segcopy_file, 
        output_file, 
        method=method, 
        manual_threshold=manual_threshold
    )
