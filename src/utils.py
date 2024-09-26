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
    plt.figure(figsize=(fig_width, fig_height))
    for i, result in enumerate(results):
        plt.semilogy(result['history_loss'], label=result['name'], color=upenn_colors[i % len(upenn_colors)], 
                     marker=markers[i % len(markers)], markevery=max(1, len(result['history_loss'])//20),
                     linewidth=2.5, markersize=8)
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Function gap', fontsize=16)
    # plt.title(f'Function gap')
    plt.legend(loc='best', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/loss_{experiment_name}.pdf")
    plt.show()

    # Plot distance to optimal solution
    plt.figure(figsize=(fig_width, fig_height))
    for i, result in enumerate(results):
        if 'history_dist_to_opt_solution' in result and len(result['history_dist_to_opt_solution']) > 0:
            plt.semilogy(result['history_dist_to_opt_solution'], label=result['name'], color=upenn_colors[i % len(upenn_colors)], 
                         marker=markers[i % len(markers)], markevery=max(1, len(result['history_dist_to_opt_solution'])//20),
                         linewidth=2.5, markersize=8)
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Distance to optimal solution', fontsize=16)
    # plt.title(f'Distance to Optimal Solution')
    plt.legend(loc='best', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/distance_to_optimal_{experiment_name}.pdf")
    plt.show()

    # Plot step size list
    plt.figure(figsize=(fig_width, fig_height))
    for i, result in enumerate(results):
        plt.semilogy(result['step_size_list'], label=result['name'], color=upenn_colors[i % len(upenn_colors)], 
                     marker=markers[i % len(markers)], markevery=max(1, len(result['step_size_list'])//20),
                     linewidth=2.5, markersize=8)
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Stepsize', fontsize=16)
    # plt.title(f'Step Size')
    plt.legend(loc='best', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/step_size_{experiment_name}.pdf")
    plt.show()


    if plot_smallest_so_far:
        # Plot smallest loss seen so far
        plt.figure(figsize=(fig_width, fig_height))
        for i, result in enumerate(results):
            smallest_loss = np.minimum.accumulate(result['history_loss'])
            plt.semilogy(smallest_loss, label=f"{result['name']} (Smallest)", color=upenn_colors[i % len(upenn_colors)], 
                         marker=markers[i % len(markers)], markevery=max(1, len(smallest_loss)//20),
                         linewidth=2.5, markersize=8, linestyle='--')
        plt.xlabel('Iterations', fontsize=16)
        plt.ylabel('Smallest Loss Seen (log scale)', fontsize=16)
        plt.title(f'Smallest Loss Seen So Far for GDPolyak Cases - {experiment_name}')
        plt.legend(loc='best', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/smallest_loss_{experiment_name}.pdf")
        plt.show()

        # Plot smallest distance to optimal solution seen so far
        plt.figure(figsize=(fig_width, fig_height))
        for i, result in enumerate(results):
            if 'history_dist_to_opt_solution' in result and len(result['history_dist_to_opt_solution']) > 0:
                smallest_dist = np.minimum.accumulate(result['history_dist_to_opt_solution'])
                plt.semilogy(smallest_dist, label=f"{result['name']} (Smallest)", color=upenn_colors[i % len(upenn_colors)], 
                             marker=markers[i % len(markers)], markevery=max(1, len(smallest_dist)//20),
                             linewidth=2.5, markersize=8, linestyle='--')
        plt.xlabel('Iterations', fontsize=16)
        plt.ylabel('Smallest Distance to Optimal Solution (log scale)', fontsize=16)
        plt.title(f'Smallest Distance to Optimal Solution Seen So Far for GDPolyak Cases - {experiment_name}')
        plt.legend(loc='best', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/smallest_distance_to_optimal_{experiment_name}.pdf")
        plt.show()