"""
Generate plots for evaluation results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config
from utils.logger import ensure_project_dirs, get_logger


def plot_results(episode_trace):
    """
    Generate plots from episode trace
    
    Args:
        episode_trace: List of dictionaries with episode data
    """
    logger = get_logger("evaluation.plots")
    if episode_trace is None or len(episode_trace) == 0:
        logger.info("No episode trace data available for plotting")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(episode_trace)
    
    # Create output directory
    ensure_project_dirs()
    
    # Plot 1: Grid Import vs PV Generation
    plt.figure(figsize=(12, 6))
    hours = df['hour'].values
    plt.plot(hours, df['grid_import'].values, 'b-', label='Grid Import', linewidth=2)
    plt.plot(hours, df['pv'].values, 'g--', label='PV Generation', linewidth=2)
    plt.fill_between(hours, 0, df['grid_import'].values, alpha=0.3, color='blue')
    plt.fill_between(hours, 0, df['pv'].values, alpha=0.2, color='green')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Power (kW)', fontsize=12)
    plt.title('Grid Import vs PV Generation Over 24 Hours', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 23)
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    grid_import_path = str(config.RESULTS_PLOTS_DIR / config.PLOT_GRID_IMPORT_VS_PV_FILENAME)
    plt.savefig(grid_import_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", grid_import_path)
    
    # Plot 2: Battery SOC
    plt.figure(figsize=(12, 6))
    plt.plot(hours, df['soc'].values * 100, 'r-', label='State of Charge', linewidth=2)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% SOC')
    plt.fill_between(hours, 0, df['soc'].values * 100, alpha=0.3, color='red')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('SOC (%)', fontsize=12)
    plt.title('Battery State of Charge Over 24 Hours', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 23)
    plt.ylim(0, 100)
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    battery_soc_path = str(config.RESULTS_PLOTS_DIR / config.PLOT_BATTERY_SOC_FILENAME)
    plt.savefig(battery_soc_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", battery_soc_path)
    
    # Plot 3: Battery Health Trajectory
    plt.figure(figsize=(12, 6))
    plt.plot(hours, df['battery_health'].values, 'c-', label='Battery Health', linewidth=2)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.6, label='Initial Health')
    plt.axhline(y=config.BATTERY_MIN_HEALTH, color='red', linestyle='--', alpha=0.7, label='Minimum Health Bound')
    plt.fill_between(hours, df['battery_health'].values, 1.0, alpha=0.15, color='cyan')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Battery Health (0-1)', fontsize=12)
    plt.title('Battery Health Trajectory Over 24 Hours', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 23)
    plt.ylim(config.BATTERY_MIN_HEALTH, 1.01)
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    battery_health_path = str(config.RESULTS_PLOTS_DIR / config.PLOT_BATTERY_HEALTH_FILENAME)
    plt.savefig(battery_health_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", battery_health_path)

    # Plot 4: Comfort Score
    plt.figure(figsize=(12, 6))
    plt.plot(hours, df['comfort'].values, 'm-', label='Comfort Score', linewidth=2)
    plt.axhline(y=0.60, color='red', linestyle='--', alpha=0.7, label='Hard Constraint (0.60)', linewidth=2)
    plt.axhline(y=0.80, color='orange', linestyle='--', alpha=0.7, label='Violation Threshold (0.80)', linewidth=1.5)
    plt.fill_between(hours, 0, df['comfort'].values, alpha=0.3, color='magenta')
    plt.fill_between(hours, 0, 0.60, alpha=0.2, color='red')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Comfort Score', fontsize=12)
    plt.title('Consumer Comfort Score Over 24 Hours', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 23)
    plt.ylim(0, 1.0)
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    comfort_score_path = str(config.RESULTS_PLOTS_DIR / config.PLOT_COMFORT_SCORE_FILENAME)
    plt.savefig(comfort_score_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", comfort_score_path)
    
    # Plot 5: Combined Overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Grid Import vs PV
    ax1 = axes[0, 0]
    ax1.plot(hours, df['grid_import'].values, 'b-', label='Grid Import', linewidth=2)
    ax1.plot(hours, df['pv'].values, 'g--', label='PV Generation', linewidth=2)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Grid Import vs PV')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 23)
    
    # SOC
    ax2 = axes[0, 1]
    ax2.plot(hours, df['soc'].values * 100, 'r-', linewidth=2)
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('SOC (%)')
    ax2.set_title('Battery State of Charge')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 23)
    ax2.set_ylim(0, 100)
    
    # Comfort
    ax3 = axes[1, 0]
    ax3.plot(hours, df['comfort'].values, 'm-', linewidth=2)
    ax3.axhline(y=0.60, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Comfort')
    ax3.set_title('Comfort Score')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 23)
    ax3.set_ylim(0, 1.0)
    
    # Actions
    ax4 = axes[1, 1]
    action_names = ['Do Nothing', 'Discharge', 'Charge', 'Shift Load', 'Comfort Sacrifice']
    action_colors = ['gray', 'blue', 'green', 'orange', 'red']
    for action_id in range(5):
        action_mask = df['action'].values == action_id
        if np.any(action_mask):
            ax4.scatter(df['hour'].values[action_mask], 
                       [action_id] * np.sum(action_mask),
                       c=action_colors[action_id], 
                       label=action_names[action_id],
                       alpha=0.6, s=50)
    ax4.set_xlabel('Hour')
    ax4.set_ylabel('Action')
    ax4.set_title('Actions Taken')
    ax4.set_yticks(range(5))
    ax4.set_yticklabels(action_names, fontsize=8)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 23)
    
    plt.tight_layout()
    overview_path = str(config.RESULTS_PLOTS_DIR / config.PLOT_OVERVIEW_FILENAME)
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", overview_path)
    
    logger.info("All plots generated successfully")

