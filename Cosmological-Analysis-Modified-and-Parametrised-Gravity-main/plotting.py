# plotting.py
"""
All figure-generation routines: dark-energy potential reconstruction,
omega-component evolution tracks, and corner plots.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import corner

import parameterisations


def plot_dark_energy_potential(flat_chain, model_name, potential_func, output_dir, run_mode, max_z=2.0):
    if potential_func is None:
        return
    print("Reconstructing Dark Energy Potential V(phi)/rho_de,0...")
    z_grid = np.linspace(0.0, max_z, 200)
    sample_indices = np.random.choice(flat_chain.shape[0], size=200, replace=False)
    potential_curves = []
    for idx in sample_indices:
        params = flat_chain[idx]
        v_z = [potential_func(zi, tuple(params[1:])) for zi in z_grid]
        potential_curves.append(v_z)

    potential_curves = np.array(potential_curves)
    plt.figure(figsize=(7, 5))
    plt.plot(z_grid, np.median(potential_curves, axis=0), color='teal', lw=2,
              label=r'Best-fit $V(\phi)/\rho_{\rm de,0}$')
    plt.fill_between(z_grid, np.percentile(potential_curves, 16, axis=0),
                      np.percentile(potential_curves, 84, axis=0), color='teal', alpha=0.3)
    plt.xlabel(r'Redshift ($z$)', fontsize=12)
    plt.ylabel(r'$V(\phi) / \rho_{\rm de,0}$', fontsize=12)
    plt.title(f'Scalar Field Potential Reconstruction: {model_name}', fontsize=13)
    plt.xlim(0, max_z)
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_filename = os.path.join(output_dir, f"potential_{model_name}_{run_mode}.pdf")
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()


def plot_and_save_omega_components(flat_chain, chosen_model_name, output_dir, run_mode,
                                    fixed_delta, fixed_alpha):
    print(f"\nEvaluating Density Component Evolution Path for Model: [{chosen_model_name}]...")
    max_z = 2.4
    z_grid = np.linspace(0.0, max_z, 100)
    sample_indices = np.random.choice(flat_chain.shape[0], size=200, replace=False)

    omega1_curves = []
    omega2_curves = []

    for idx in sample_indices:
        params_sample = flat_chain[idx]
        if chosen_model_name == "Custom_Universal":
            Omega_0_val = params_sample[1]
            delta_val = fixed_delta
            alpha_val = fixed_alpha
        elif chosen_model_name == "param_Hubble_fixed_delta":
            Omega_0_val = params_sample[1]
            alpha_val = params_sample[2]
            delta_val = fixed_delta
        elif chosen_model_name == "param_Hubble_fixed_alpha":
            Omega_0_val = params_sample[1]
            delta_val = params_sample[2]
            alpha_val = fixed_alpha
        else:
            return

        o1_z, o2_z = [], []
        for zi in z_grid:
            o1, o2 = parameterisations.calculate_omega_components(zi, Omega_0_val, delta_val, alpha_val)
            o1_z.append(o1)
            o2_z.append(o2)
        omega1_curves.append(o1_z)
        omega2_curves.append(o2_z)

    omega1_curves = np.array(omega1_curves)
    omega2_curves = np.array(omega2_curves)

    m1, l1, h1 = np.median(omega1_curves, axis=0), np.percentile(omega1_curves, 16, axis=0), np.percentile(omega1_curves, 84, axis=0)
    m2, l2, h2 = np.median(omega2_curves, axis=0), np.percentile(omega2_curves, 16, axis=0), np.percentile(omega2_curves, 84, axis=0)

    print(f"    {'Redshift z':12s} | {'Omega_1 (Matter-like)':22s} | {'Omega_2 (DE-like)':22s}")
    print("    " + "-" * 65)
    for target_z in [0.0, 0.5, 1.0, 2.0]:
        z_idx = np.abs(z_grid - target_z).argmin()
        print(f"    {z_grid[z_idx]:12.2f} | {m1[z_idx]:.4f} (+{h1[z_idx]-m1[z_idx]:.4f}/-{m1[z_idx]-l1[z_idx]:.4f}) | "
              f"{m2[z_idx]:.4f} (+{h2[z_idx]-m2[z_idx]:.4f}/-{m2[z_idx]-l2[z_idx]:.4f})")

    plt.figure(figsize=(7, 5))
    plt.plot(z_grid, m1, color='crimson', lw=2, label=r'$\Omega_1(z)$ [Matter-like Component]')
    plt.fill_between(z_grid, l1, h1, color='crimson', alpha=0.15)
    plt.plot(z_grid, m2, color='royalblue', lw=2, label=r'$\Omega_2(z)$ [DE-like Component]')
    plt.fill_between(z_grid, l2, h2, color='royalblue', alpha=0.15)
    plt.xlabel(r'Redshift ($z$)', fontsize=12)
    plt.ylabel(r'Component Density Scale ($\Omega_x$)', fontsize=12)
    plt.title(f'Evolutionary Profile: {chosen_model_name}', fontsize=12)
    plt.xlim(0, max_z)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)

    plot_filename = os.path.join(output_dir, f"omega_components_{chosen_model_name}_{run_mode}.pdf")
    plt.savefig(plot_filename, bbox_inches="tight")
    print(f"Component track plot successfully saved to: {plot_filename}")
    plt.close()


def plot_corner(flat_chain, labels, chosen_model_name, output_dir, run_mode):
    fig = corner.corner(flat_chain, labels=labels, show_titles=True, quantiles=[0.16, 0.5, 0.84])
    fig.suptitle(f"Posterior Framework: {chosen_model_name}", y=1.02)

    corner_save_path = os.path.join(output_dir, f"corner_{chosen_model_name}_{run_mode}.pdf")
    fig.savefig(corner_save_path, bbox_inches="tight")
    print(f"Corner plot saved successfully inside isolated subfolder: {corner_save_path}")
    return fig


def generate_model_specific_plots(flat_chain, chosen_model_name, model_config, output_dir, run_mode,
                                   fixed_delta, fixed_alpha):
    """
    Dispatches to either the omega-component tracker (for the custom
    parametrized-Hubble models) or the dark-energy potential reconstruction
    (for the scalar-field-style models), matching the branch in the
    original main.py.
    """
    custom_targets = ["Custom_Universal", "param_Hubble_fixed_delta", "param_Hubble_fixed_alpha"]
    if chosen_model_name in custom_targets:
        plot_and_save_omega_components(flat_chain, chosen_model_name, output_dir, run_mode,
                                        fixed_delta, fixed_alpha)
    else:
        plot_dark_energy_potential(flat_chain, chosen_model_name, model_config["potential_func"],
                                    output_dir, run_mode)