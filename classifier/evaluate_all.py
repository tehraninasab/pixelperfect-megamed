#!/usr/bin/env python
"""
Analyze CheXpert classifier performance per disease across different image sizes.
This script generates detailed metrics and visualizations for each disease
in the CheXpert dataset for a specific model architecture, with separate
analysis for pretrained vs non-pretrained models.
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# Target columns (diseases)
TARGET_COLS = ["Cardiomegaly", "Lung Opacity", "Edema", "No Finding", "Pneumothorax", "Pleural Effusion"]

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze CheXpert results per disease')
    parser.add_argument('--model', type=str, default='efficientnet-b0',
                        help='Model architecture to analyze')
    parser.add_argument('--results_dir', type=str, default='results_train_synth_val_synth_reversed',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='analysis_results_train_synth_val_synth_reversed',
                        help='Directory to save analysis results')
    parser.add_argument('--separate_pretrained', action='store_true', default=True,
                        help='Create separate analyses for pretrained vs non-pretrained models')
    return parser.parse_args()

def find_metrics_files(results_dir, model):
    """Find all test_metrics.csv files for the specified model."""
    metrics_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "test_metrics.csv" and model in root:
                metrics_files.append(os.path.join(root, file))
    return metrics_files

def extract_config_info(file_path):
    """Extract model, image size, and pretrained info from file path."""
    folder = os.path.dirname(file_path)
    config = os.path.basename(folder)
    
    # Extract model name (usually the first part)
    parts = config.split('_')
    model_name = parts[0]
    
    # Extract image size
    image_size = None
    if len(parts) >= 2:
        try:
            image_size = int(parts[1])
        except ValueError:
            # Handle case where size might be in a different position
            try:
                image_size = int(parts[-1] if "pretrained" not in parts[-1] else parts[-2])
            except ValueError:
                print(f"Warning: Could not extract image size from {config}")
    
    # Check if pretrained
    is_pretrained = "pretrained" in config
    
    return model_name, image_size, is_pretrained

def collect_metrics(metrics_files, model, target_cols):
    """Collect metrics for each disease across different image sizes."""
    all_metrics = []
    
    for file_path in metrics_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping")
            continue
        
        # Extract configuration info
        model_name, image_size, is_pretrained = extract_config_info(file_path)
        
        # Skip if this is not the model we're interested in or couldn't extract image size
        if model_name != model or image_size is None:
            continue
        
        # Read metrics file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        # Extract per-disease metrics
        for _, row in df.iterrows():
            if row['Class'] in target_cols:
                metric_dict = {
                    'Disease': row['Class'],
                    'Model': model_name,
                    'Image Size': image_size,
                    'Pretrained': is_pretrained
                }
                
                # Add all other columns as metrics
                for column in df.columns:
                    if column != 'Class':
                        metric_dict[column] = row[column]
                
                all_metrics.append(metric_dict)
    
    return all_metrics

def create_markdown_report(metrics_df, model, output_dir, target_cols, pretrained_suffix=""):
    """Create a detailed markdown report with tables and findings."""
    pretrained_text = " (Pretrained)" if "pretrained" in pretrained_suffix else " (Non-pretrained)" if "non_pretrained" in pretrained_suffix else ""
    report_path = os.path.join(output_dir, f"{model}_per_disease_summary{pretrained_suffix}.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Per-Disease Performance Analysis for {model}{pretrained_text}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report analyzes the performance of the CheXpert classifier ")
        f.write(f"using the {model} architecture{pretrained_text.lower()} across different image sizes ")
        f.write("for each disease in the dataset.\n\n")
        
        # Get unique image sizes
        image_sizes = sorted(metrics_df['Image Size'].unique())
        
        f.write(f"Analyzed image sizes: {', '.join(map(str, image_sizes))}\n\n")
        
        f.write("## Performance Across Image Sizes\n\n")
        
        # For each disease, create a table of metrics across image sizes
        for disease in target_cols:
            disease_data = metrics_df[metrics_df['Disease'] == disease]
            
            if disease_data.empty:
                continue
            
            f.write(f"### {disease}\n\n")
            
            # Create a table header
            f.write("| Image Size | Accuracy | AUROC | Precision | Recall | F1 |\n")
            f.write("| ---------- | -------- | ----- | --------- | ------ | -- |\n")
            
            # Add rows for each image size
            for size in image_sizes:
                size_data = disease_data[disease_data['Image Size'] == size]
                
                if not size_data.empty:
                    row = size_data.iloc[0]
                    f.write(f"| {size} | {row.get('Accuracy', 'N/A'):.4f} | ")
                    
                    # Handle possible NaN in AUROC
                    auroc_val = row.get('AUROC', 'N/A')
                    if isinstance(auroc_val, float) and not np.isnan(auroc_val):
                        f.write(f"{auroc_val:.4f} | ")
                    else:
                        f.write("N/A | ")
                    
                    f.write(f"{row.get('Precision', 'N/A'):.4f} | ")
                    f.write(f"{row.get('Recall', 'N/A'):.4f} | ")
                    f.write(f"{row.get('F1', 'N/A'):.4f} |\n")
            
            f.write("\n")
            
            # Find best image size for this disease based on F1 score
            if 'F1' in disease_data.columns:
                best_idx = disease_data['F1'].idxmax()
                best_size = disease_data.loc[best_idx, 'Image Size']
                best_f1 = disease_data.loc[best_idx, 'F1']
                f.write(f"**Best image size for {disease} by F1 score**: ")
                f.write(f"{best_size} (F1: {best_f1:.4f})\n\n")
        
        f.write("## Best Image Size per Disease\n\n")
        f.write("| Disease | Best Image Size | F1 Score | AUROC |\n")
        f.write("| ------- | --------------- | -------- | ----- |\n")
        
        for disease in target_cols:
            disease_data = metrics_df[metrics_df['Disease'] == disease]
            
            if disease_data.empty or 'F1' not in disease_data.columns:
                continue
            
            best_idx = disease_data['F1'].idxmax()
            best_size = disease_data.loc[best_idx, 'Image Size']
            best_f1 = disease_data.loc[best_idx, 'F1']
            
            # Handle possible NaN in AUROC
            best_auroc = disease_data.loc[best_idx, 'AUROC'] if 'AUROC' in disease_data.columns else 'N/A'
            if isinstance(best_auroc, float) and np.isnan(best_auroc):
                best_auroc = 'N/A'
            
            f.write(f"| {disease} | {best_size} | {best_f1:.4f} | ")
            f.write(f"{best_auroc if isinstance(best_auroc, str) else best_auroc:.4f} |\n")
        
        f.write("\n## Conclusion\n\n")
        
        # Find overall best image size (averaging F1 across diseases)
        if 'F1' in metrics_df.columns:
            avg_f1_by_size = metrics_df.groupby('Image Size')['F1'].mean()
            overall_best_size = avg_f1_by_size.idxmax()
            
            f.write(f"Based on the average F1 score across all diseases, ")
            f.write(f"the overall best image size for {model}{pretrained_text.lower()} is **{overall_best_size}**.\n\n")
            
            f.write("However, as shown in the per-disease analysis, the optimal image size ")
            f.write("varies across different pathologies. Consider using the disease-specific ")
            f.write("optimal image sizes if focusing on particular conditions.\n")
    
    return report_path

def create_visualizations(metrics_df, model, output_dir, target_cols, pretrained_suffix=""):
    """Create visualizations for the analysis."""
    
    pretrained_text = " (Pretrained)" if "pretrained" in pretrained_suffix else " (Non-pretrained)" if "non_pretrained" in pretrained_suffix else ""
    
    # 1. F1 score by disease and image size
    plt.figure(figsize=(14, 8))
    for disease in target_cols:
        disease_data = metrics_df[metrics_df['Disease'] == disease]
        
        if not disease_data.empty and 'F1' in disease_data.columns:
            # Sort by image size for proper line plotting
            disease_data = disease_data.sort_values('Image Size')
            plt.plot(disease_data['Image Size'], disease_data['F1'], 'o-', label=disease, linewidth=2, markersize=6)

    plt.title(f'F1 Score by Disease and Image Size ({model}{pretrained_text})', fontsize=16)
    plt.xlabel('Image Size', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(sorted(metrics_df['Image Size'].unique()))
    
    # Add value labels to points
    for disease in target_cols:
        disease_data = metrics_df[metrics_df['Disease'] == disease]
        if not disease_data.empty and 'F1' in disease_data.columns:
            disease_data = disease_data.sort_values('Image Size')
            for i, row in disease_data.iterrows():
                plt.annotate(f"{row['F1']:.3f}", 
                             (row['Image Size'], row['F1']),
                             textcoords="offset points",
                             xytext=(0,10),
                             ha='center',
                             fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model}_f1_by_disease{pretrained_suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. AUROC by disease and image size
    plt.figure(figsize=(14, 8))
    for disease in target_cols:
        disease_data = metrics_df[metrics_df['Disease'] == disease]
        
        if not disease_data.empty and 'AUROC' in disease_data.columns:
            # Skip NaN values
            valid_data = disease_data.dropna(subset=['AUROC']).sort_values('Image Size')
            if not valid_data.empty:
                plt.plot(valid_data['Image Size'], valid_data['AUROC'], 'o-', label=disease, linewidth=2, markersize=6)
                
                # Add value labels
                for i, row in valid_data.iterrows():
                    plt.annotate(f"{row['AUROC']:.3f}", 
                                 (row['Image Size'], row['AUROC']),
                                 textcoords="offset points",
                                 xytext=(0,10),
                                 ha='center',
                                 fontsize=9)

    plt.title(f'AUROC by Disease and Image Size ({model}{pretrained_text})', fontsize=16)
    plt.xlabel('Image Size', fontsize=14)
    plt.ylabel('AUROC', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(sorted(metrics_df['Image Size'].unique()))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model}_auroc_by_disease{pretrained_suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap of F1 scores across diseases and image sizes
    if 'F1' in metrics_df.columns:
        # Create pivot table - rows: diseases, columns: image sizes, values: F1 scores
        pivot_df = metrics_df.pivot_table(
            index='Disease', 
            columns='Image Size', 
            values='F1', 
            aggfunc='first'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.4f', linewidths=.5, cbar_kws={'label': 'F1 Score'})
        plt.title(f'F1 Score Heatmap by Disease and Image Size ({model}{pretrained_text})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_f1_heatmap{pretrained_suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Bar chart for best image size frequency
    if 'F1' in metrics_df.columns:
        best_sizes = []
        
        for disease in target_cols:
            disease_data = metrics_df[metrics_df['Disease'] == disease]
            
            if not disease_data.empty:
                best_idx = disease_data['F1'].idxmax()
                best_sizes.append({
                    'Disease': disease,
                    'Best Size': disease_data.loc[best_idx, 'Image Size'],
                    'F1 Score': disease_data.loc[best_idx, 'F1']
                })
        
        if best_sizes:
            best_df = pd.DataFrame(best_sizes)
            
            # Count occurrences of each best size
            size_counts = best_df['Best Size'].value_counts().reset_index()
            size_counts.columns = ['Image Size', 'Count']
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(size_counts['Image Size'].astype(str), size_counts['Count'], color='tab:blue', alpha=0.7)
            plt.title(f'Frequency of Best Image Size Across Diseases ({model}{pretrained_text})', fontsize=16)
            plt.xlabel('Image Size', fontsize=14)
            plt.ylabel('Number of Diseases', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f"{int(height)}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model}_best_size_frequency{pretrained_suffix}.png"), dpi=300, bbox_inches='tight')
            plt.close()

def create_comparison_visualization(pretrained_df, non_pretrained_df, model, output_dir, target_cols):
    """Create comparison visualizations between pretrained and non-pretrained models."""
    
    # 1. Side-by-side F1 comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Pretrained plot
    for disease in target_cols:
        disease_data = pretrained_df[pretrained_df['Disease'] == disease]
        if not disease_data.empty and 'F1' in disease_data.columns:
            disease_data = disease_data.sort_values('Image Size')
            ax1.plot(disease_data['Image Size'], disease_data['F1'], 'o-', label=disease, linewidth=2, markersize=6)
    
    ax1.set_title(f'F1 Score - Pretrained {model}', fontsize=14)
    ax1.set_xlabel('Image Size', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(sorted(pretrained_df['Image Size'].unique()))
    
    # Non-pretrained plot
    for disease in target_cols:
        disease_data = non_pretrained_df[non_pretrained_df['Disease'] == disease]
        if not disease_data.empty and 'F1' in disease_data.columns:
            disease_data = disease_data.sort_values('Image Size')
            ax2.plot(disease_data['Image Size'], disease_data['F1'], 'o-', label=disease, linewidth=2, markersize=6)
    
    ax2.set_title(f'F1 Score - Non-pretrained {model}', fontsize=14)
    ax2.set_xlabel('Image Size', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(sorted(non_pretrained_df['Image Size'].unique()))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model}_f1_comparison_pretrained_vs_non.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Direct comparison plot (overlapping lines with different styles)
    plt.figure(figsize=(16, 10))
    
    for disease in target_cols:
        # Pretrained data
        pretrained_disease = pretrained_df[pretrained_df['Disease'] == disease]
        if not pretrained_disease.empty and 'F1' in pretrained_disease.columns:
            pretrained_disease = pretrained_disease.sort_values('Image Size')
            plt.plot(pretrained_disease['Image Size'], pretrained_disease['F1'], 
                    'o-', label=f'{disease} (Pretrained)', linewidth=2, markersize=8, alpha=0.8)
        
        # Non-pretrained data
        non_pretrained_disease = non_pretrained_df[non_pretrained_df['Disease'] == disease]
        if not non_pretrained_disease.empty and 'F1' in non_pretrained_disease.columns:
            non_pretrained_disease = non_pretrained_disease.sort_values('Image Size')
            plt.plot(non_pretrained_disease['Image Size'], non_pretrained_disease['F1'], 
                    's--', label=f'{disease} (Non-pretrained)', linewidth=2, markersize=8, alpha=0.8)
    
    plt.title(f'F1 Score Comparison: Pretrained vs Non-pretrained {model}', fontsize=16)
    plt.xlabel('Image Size', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Get all unique image sizes from both datasets
    all_sizes = sorted(set(pretrained_df['Image Size'].unique()) | set(non_pretrained_df['Image Size'].unique()))
    plt.xticks(all_sizes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model}_f1_overlay_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance improvement heatmap
    improvement_data = []
    
    for disease in target_cols:
        pretrained_disease = pretrained_df[pretrained_df['Disease'] == disease]
        non_pretrained_disease = non_pretrained_df[non_pretrained_df['Disease'] == disease]
        
        # Get common image sizes
        if not pretrained_disease.empty and not non_pretrained_disease.empty:
            pretrained_sizes = set(pretrained_disease['Image Size'].unique())
            non_pretrained_sizes = set(non_pretrained_disease['Image Size'].unique())
            common_sizes = pretrained_sizes & non_pretrained_sizes
            
            for size in common_sizes:
                pretrained_f1 = pretrained_disease[pretrained_disease['Image Size'] == size]['F1'].iloc[0]
                non_pretrained_f1 = non_pretrained_disease[non_pretrained_disease['Image Size'] == size]['F1'].iloc[0]
                
                improvement = pretrained_f1 - non_pretrained_f1
                improvement_data.append({
                    'Disease': disease,
                    'Image Size': size,
                    'F1 Improvement': improvement
                })
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        pivot_improvement = improvement_df.pivot_table(
            index='Disease', 
            columns='Image Size', 
            values='F1 Improvement', 
            aggfunc='first'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_improvement, annot=True, cmap='RdBu_r', center=0, fmt='.4f', 
                   linewidths=.5, cbar_kws={'label': 'F1 Improvement (Pretrained - Non-pretrained)'})
        plt.title(f'F1 Score Improvement with Pretraining ({model})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_f1_improvement_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()

def create_disease_csv_files(metrics_df, model, output_dir, target_cols, pretrained_suffix=""):
    """Create individual CSV files for each disease."""
    for disease in target_cols:
        disease_data = metrics_df[metrics_df['Disease'] == disease]
        
        if not disease_data.empty:
            # Keep only relevant columns and sort by image size
            columns_to_keep = ['Image Size', 'Accuracy', 'AUROC', 'Precision', 'Recall', 'F1']
            csv_data = disease_data[columns_to_keep].sort_values('Image Size')
            
            # Save to CSV
            csv_path = os.path.join(output_dir, f"{model}_{disease}{pretrained_suffix}.csv")
            csv_data.to_csv(csv_path, index=False)
            print(f"Created CSV for {disease}: {csv_path}")

def create_comparison_csv(metrics_df, model, output_dir, target_cols, pretrained_suffix=""):
    """Create a consolidated comparison CSV across diseases."""
    # Define image sizes to include
    image_sizes = sorted(metrics_df['Image Size'].unique())
    
    # Create header row
    header = ["Disease", "Metric"] + [f"{size}px" for size in image_sizes]
    
    # Create data rows
    rows = []
    for disease in target_cols:
        disease_data = metrics_df[metrics_df['Disease'] == disease]
        
        if disease_data.empty:
            continue
        
        # For each metric
        for metric in ['Accuracy', 'AUROC', 'Precision', 'Recall', 'F1']:
            row = [disease, metric]
            
            # For each image size
            for size in image_sizes:
                size_data = disease_data[disease_data['Image Size'] == size]
                
                if not size_data.empty and metric in size_data.columns:
                    value = size_data.iloc[0][metric]
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
            
            rows.append(row)
    
    # Create and save the comparison CSV
    csv_path = os.path.join(output_dir, f"{model}_disease_comparison{pretrained_suffix}.csv")
    with open(csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(map(str, row)) + '\n')
    
    print(f"Created comparison CSV: {csv_path}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing {args.model} in {args.results_dir}, saving to {args.output_dir}")
    
    # Find all metrics files for the model
    metrics_files = find_metrics_files(args.results_dir, args.model)
    print(f"Found {len(metrics_files)} metrics files")
    
    if not metrics_files:
        print(f"Error: No metrics files found for model {args.model}")
        return
    
    # Collect metrics
    all_metrics = collect_metrics(metrics_files, args.model, TARGET_COLS)
    
    if not all_metrics:
        print(f"Error: No valid metrics found for model {args.model}")
        return
    
    # Create dataframe
    metrics_df = pd.DataFrame(all_metrics)
    print(f"Collected metrics for {len(metrics_df)} configurations")
    
    # Save raw data
    raw_data_path = os.path.join(args.output_dir, f"{args.model}_per_disease_metrics_all.csv")
    metrics_df.to_csv(raw_data_path, index=False)
    print(f"Saved raw metrics to {raw_data_path}")
    
    if args.separate_pretrained:
        # Separate pretrained and non-pretrained models
        pretrained_df = metrics_df[metrics_df['Pretrained'] == True]
        non_pretrained_df = metrics_df[metrics_df['Pretrained'] == False]
        
        print(f"Pretrained models: {len(pretrained_df)} configurations")
        print(f"Non-pretrained models: {len(non_pretrained_df)} configurations")
        
        # Create separate analyses for pretrained models
        if not pretrained_df.empty:
            print("\n=== Analyzing Pretrained Models ===")
            
            # Save pretrained raw data
            pretrained_raw_path = os.path.join(args.output_dir, f"{args.model}_per_disease_metrics_pretrained.csv")
            pretrained_df.to_csv(pretrained_raw_path, index=False)
            
            # Create markdown report
            report_path = create_markdown_report(pretrained_df, args.model, args.output_dir, TARGET_COLS, "_pretrained")
            print(f"Created pretrained markdown report: {report_path}")
            
            # Create visualizations
            create_visualizations(pretrained_df, args.model, args.output_dir, TARGET_COLS, "_pretrained")
            print(f"Created pretrained visualizations")
            
            # Create CSV files
            create_disease_csv_files(pretrained_df, args.model, args.output_dir, TARGET_COLS, "_pretrained")
            create_comparison_csv(pretrained_df, args.model, args.output_dir, TARGET_COLS, "_pretrained")
        
        # Create separate analyses for non-pretrained models
        if not non_pretrained_df.empty:
            print("\n=== Analyzing Non-Pretrained Models ===")
            
            # Save non-pretrained raw data
            non_pretrained_raw_path = os.path.join(args.output_dir, f"{args.model}_per_disease_metrics_non_pretrained.csv")
            non_pretrained_df.to_csv(non_pretrained_raw_path, index=False)
            
            # Create markdown report
            report_path = create_markdown_report(non_pretrained_df, args.model, args.output_dir, TARGET_COLS, "_non_pretrained")
            print(f"Created non-pretrained markdown report: {report_path}")
            
            # Create visualizations
            create_visualizations(non_pretrained_df, args.model, args.output_dir, TARGET_COLS, "_non_pretrained")
            print(f"Created non-pretrained visualizations")
            
            # Create CSV files
            create_disease_csv_files(non_pretrained_df, args.model, args.output_dir, TARGET_COLS, "_non_pretrained")
            create_comparison_csv(non_pretrained_df, args.model, args.output_dir, TARGET_COLS, "_non_pretrained")
        
        # Create comparison visualizations if both exist
        if not pretrained_df.empty and not non_pretrained_df.empty:
            print("\n=== Creating Comparison Visualizations ===")
            create_comparison_visualization(pretrained_df, non_pretrained_df, args.model, args.output_dir, TARGET_COLS)
            print("Created comparison visualizations")
    
    else:
        # Original combined analysis
        print("\n=== Creating Combined Analysis ===")
        
        # Create markdown report
        report_path = create_markdown_report(metrics_df, args.model, args.output_dir, TARGET_COLS)
        print(f"Created markdown report: {report_path}")
        
        # Create visualizations
        create_visualizations(metrics_df, args.model, args.output_dir, TARGET_COLS)
        print(f"Created visualizations in {args.output_dir}")
        
        # Create individual disease CSV files
        create_disease_csv_files(metrics_df, args.model, args.output_dir, TARGET_COLS)
        
        # Create comparison CSV
        create_comparison_csv(metrics_df, args.model, args.output_dir, TARGET_COLS)
    
    # Create comprehensive summary file
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"======================================\n")
        f.write(f"{args.model} Per-Disease Performance Summary\n")
        f.write(f"======================================\n\n")
        
        if args.separate_pretrained:
            f.write("SEPARATE ANALYSIS MODE\n")
            f.write("The analysis has been separated into pretrained and non-pretrained models.\n\n")
            
            f.write("Files created:\n\n")
            f.write("RAW DATA:\n")
            f.write(f"- {args.model}_per_disease_metrics_all.csv (All configurations)\n")
            f.write(f"- {args.model}_per_disease_metrics_pretrained.csv (Pretrained models only)\n")
            f.write(f"- {args.model}_per_disease_metrics_non_pretrained.csv (Non-pretrained models only)\n\n")
            
            f.write("REPORTS:\n")
            f.write(f"- {args.model}_per_disease_summary_pretrained.md (Pretrained analysis)\n")
            f.write(f"- {args.model}_per_disease_summary_non_pretrained.md (Non-pretrained analysis)\n\n")
            
            f.write("INDIVIDUAL DISEASE CSVs:\n")
            f.write("Pretrained models:\n")
            for disease in TARGET_COLS:
                f.write(f"  - {args.model}_{disease}_pretrained.csv\n")
            f.write("\nNon-pretrained models:\n")
            for disease in TARGET_COLS:
                f.write(f"  - {args.model}_{disease}_non_pretrained.csv\n")
            
            f.write(f"\nCOMPARISON CSVs:\n")
            f.write(f"- {args.model}_disease_comparison_pretrained.csv\n")
            f.write(f"- {args.model}_disease_comparison_non_pretrained.csv\n\n")
            
            f.write("VISUALIZATIONS:\n")
            f.write("Pretrained models:\n")
            f.write(f"  - {args.model}_f1_by_disease_pretrained.png\n")
            f.write(f"  - {args.model}_auroc_by_disease_pretrained.png\n")
            f.write(f"  - {args.model}_f1_heatmap_pretrained.png\n")
            f.write(f"  - {args.model}_best_size_frequency_pretrained.png\n")
            
            f.write("\nNon-pretrained models:\n")
            f.write(f"  - {args.model}_f1_by_disease_non_pretrained.png\n")
            f.write(f"  - {args.model}_auroc_by_disease_non_pretrained.png\n")
            f.write(f"  - {args.model}_f1_heatmap_non_pretrained.png\n")
            f.write(f"  - {args.model}_best_size_frequency_non_pretrained.png\n")
            
            f.write("\nComparison visualizations:\n")
            f.write(f"  - {args.model}_f1_comparison_pretrained_vs_non.png (side-by-side)\n")
            f.write(f"  - {args.model}_f1_overlay_comparison.png (overlapping lines)\n")
            f.write(f"  - {args.model}_f1_improvement_heatmap.png (improvement with pretraining)\n\n")
            
        else:
            f.write("COMBINED ANALYSIS MODE\n")
            f.write("The analysis includes both pretrained and non-pretrained models together.\n\n")
            
            f.write("Files created:\n\n")
            f.write(f"1. {args.model}_per_disease_metrics_all.csv - Raw metrics for all diseases and image sizes\n")
            f.write(f"2. {args.model}_per_disease_summary.md - Detailed markdown report with tables and findings\n")
            f.write("3. Individual disease CSV files:\n")
            
            for disease in TARGET_COLS:
                f.write(f"   - {args.model}_{disease}.csv\n")
            
            f.write(f"\n4. {args.model}_disease_comparison.csv - Consolidated comparison across diseases\n\n")
            f.write("5. Visualizations:\n")
            f.write(f"   - {args.model}_f1_by_disease.png - F1 scores by disease and image size\n")
            f.write(f"   - {args.model}_auroc_by_disease.png - AUROC by disease and image size\n")
            f.write(f"   - {args.model}_f1_heatmap.png - Heatmap of F1 scores\n")
            f.write(f"   - {args.model}_best_size_frequency.png - Frequency of best image sizes\n\n")
        
        f.write(f"Analysis completed successfully!\n")
        f.write(f"Check the markdown reports for detailed insights.\n")
    
    print(f"Created summary file: {summary_path}")
    print(f"Analysis complete! All files saved to {args.output_dir}/")
    
    # Print quick summary to console
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    if args.separate_pretrained:
        pretrained_count = len(metrics_df[metrics_df['Pretrained'] == True])
        non_pretrained_count = len(metrics_df[metrics_df['Pretrained'] == False])
        
        print(f"Model analyzed: {args.model}")
        print(f"Total configurations: {len(metrics_df)}")
        print(f"  - Pretrained: {pretrained_count}")
        print(f"  - Non-pretrained: {non_pretrained_count}")
        print(f"Diseases analyzed: {len(TARGET_COLS)}")
        print(f"Image sizes: {sorted(metrics_df['Image Size'].unique())}")
        
        if pretrained_count > 0 and non_pretrained_count > 0:
            print(f"\n✓ Separate analyses created for pretrained and non-pretrained models")
            print(f"✓ Comparison visualizations created")
        
        print(f"\nKey files to check:")
        print(f"  - Pretrained report: {args.model}_per_disease_summary_pretrained.md")
        print(f"  - Non-pretrained report: {args.model}_per_disease_summary_non_pretrained.md")
        print(f"  - Comparison plots: {args.model}_f1_comparison_*.png")
        
    else:
        print(f"Model analyzed: {args.model}")
        print(f"Total configurations: {len(metrics_df)}")
        print(f"Diseases analyzed: {len(TARGET_COLS)}")
        print(f"Image sizes: {sorted(metrics_df['Image Size'].unique())}")
        print(f"\nKey file to check:")
        print(f"  - Main report: {args.model}_per_disease_summary.md")

if __name__ == "__main__":
    main()