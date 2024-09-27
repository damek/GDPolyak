import matplotlib.pyplot as plt
import numpy as np
import os


def plot_gdpolyak_results(results, fig_width=7, fig_height=7, plot_smallest_so_far=False, save_path=None, experiment_name=""):
    # Set up the plot style
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
    })
    plt.rcParams.update({'font.size': 14})

    # UPenn colors and markers
    upenn_colors = ['#011F5B', '#990000', '#82C0E9', '#F2C100', '#5D1E1E']
    markers = ['o', 's', '^', 'D', 'v']

    # Create save_path directory if it doesn't exist
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Plot loss
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for i, result in enumerate(results):
        ax.semilogy(result['history_loss'], label=result['name'], color=upenn_colors[i % len(upenn_colors)], 
                     marker=markers[i % len(markers)], markevery=max(1, len(result['history_loss'])//20),
                     linewidth=2.5, markersize=8)
    ax.set_xlabel('Iterations', fontsize=16)
    ax.set_ylabel('Function gap', fontsize=16)
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}/loss_{experiment_name}.pdf", bbox_inches='tight')
    plt.show()

    # Plot distance to optimal solution
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for i, result in enumerate(results):
        if 'history_dist_to_opt_solution' in result and len(result['history_dist_to_opt_solution']) > 0:
            ax.semilogy(result['history_dist_to_opt_solution'], label=result['name'], color=upenn_colors[i % len(upenn_colors)], 
                         marker=markers[i % len(markers)], markevery=max(1, len(result['history_dist_to_opt_solution'])//20),
                         linewidth=2.5, markersize=8)
    ax.set_xlabel('Iterations', fontsize=16)
    ax.set_ylabel('Distance to optimal solution', fontsize=16)
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}/distance_to_optimal_{experiment_name}.pdf", bbox_inches='tight')
    plt.show()

    # Plot step size list
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for i, result in enumerate(results):
        ax.semilogy(result['step_size_list'], label=result['name'], color=upenn_colors[i % len(upenn_colors)], 
                     marker=markers[i % len(markers)], markevery=max(1, len(result['step_size_list'])//20),
                     linewidth=2.5, markersize=8)
    ax.set_xlabel('Iterations', fontsize=16)
    ax.set_ylabel('Stepsize', fontsize=16)
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}/step_size_{experiment_name}.pdf", bbox_inches='tight')
    plt.show()


    if plot_smallest_so_far:
        # Plot smallest loss seen so far
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        for i, result in enumerate(results):
            smallest_loss = np.minimum.accumulate(result['history_loss'])
            ax.semilogy(smallest_loss, label=f"{result['name']} (Smallest)", color=upenn_colors[i % len(upenn_colors)], 
                         marker=markers[i % len(markers)], markevery=max(1, len(smallest_loss)//20),
                         linewidth=2.5, markersize=8, linestyle='--')
        ax.set_xlabel('Iterations', fontsize=16)
        ax.set_ylabel('Smallest Loss Seen (log scale)', fontsize=16)
        ax.set_title(f'Smallest Loss Seen So Far for GDPolyak Cases - {experiment_name}')
        ax.legend(loc='best', fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.tight_layout()
        if save_path:
            fig.savefig(f"{save_path}/smallest_loss_{experiment_name}.pdf", bbox_inches='tight')
        plt.show()

        # Plot smallest distance to optimal solution seen so far
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        for i, result in enumerate(results):
            if 'history_dist_to_opt_solution' in result and len(result['history_dist_to_opt_solution']) > 0:
                smallest_dist = np.minimum.accumulate(result['history_dist_to_opt_solution'])
                ax.semilogy(smallest_dist, label=f"{result['name']} (Smallest)", color=upenn_colors[i % len(upenn_colors)], 
                             marker=markers[i % len(markers)], markevery=max(1, len(smallest_dist)//20),
                             linewidth=2.5, markersize=8, linestyle='--')
        ax.set_xlabel('Iterations', fontsize=16)
        ax.set_ylabel('Smallest Distance to Optimal Solution (log scale)', fontsize=16)
        ax.set_title(f'Smallest Distance to Optimal Solution Seen So Far for GDPolyak Cases - {experiment_name}')
        ax.legend(loc='best', fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.tight_layout()
        if save_path:
            fig.savefig(f"{save_path}/smallest_distance_to_optimal_{experiment_name}.pdf", bbox_inches='tight')
        plt.show()