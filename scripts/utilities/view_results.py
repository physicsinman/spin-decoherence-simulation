#!/usr/bin/env python3
"""
Quick script to view all result plots in a grid.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

def view_all_results(results_dir='results'):
    """Display all PNG plots from results directory in a grid."""
    # Find all PNG files
    png_files = sorted(glob.glob(os.path.join(results_dir, '*.png')))
    
    if not png_files:
        print(f"No PNG files found in {results_dir}")
        return
    
    print(f"Found {len(png_files)} plot files:")
    for f in png_files:
        print(f"  - {os.path.basename(f)}")
    
    # Create grid layout
    n_files = len(png_files)
    n_cols = 2
    n_rows = (n_files + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows))
    if n_files == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, png_file in enumerate(png_files):
        img = mpimg.imread(png_file)
        axes[i].imshow(img)
        axes[i].set_title(os.path.basename(png_file), fontsize=10)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(n_files, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("\nGraphs displayed. Close the window when done.")

if __name__ == '__main__':
    view_all_results()

