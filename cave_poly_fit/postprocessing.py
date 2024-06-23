from utils import ExperimentDetails
import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_derivative(fn_values: np.array, ord: int):
    """
    Compute ord-th derivative of fn_values
    """
    for _ in range(ord):
        fn_values = np.gradient(fn_values)
    
    return fn_values

def postprocess_grads(details: ExperimentDetails):
    l_one_norms = []
    l_inf_norms = []
    l_inf_relative_norms = []
    MSE_vals = []
    
    made_typical_plots = False
    made_outlier_plots = False

    for experiment_idx in range(details.experiment_size):
        fit = details.fits_per_experiment[experiment_idx]
        grads = details.grads_per_experiment[experiment_idx]
        grad_range = np.max(grads) - np.min(grads)
        
        errors = []
        fit_values = []

        for idx, a_val in enumerate(details.alpha_values):
            fit_value = fit(a_val)
            errors.append(np.abs(grads[idx] - fit_value))
            fit_values.append(fit_value)

        l_one_norm = np.linalg.norm(errors, ord=1)
        l_one_norms.append(l_one_norm)
        l_inf_norm = np.linalg.norm(errors, ord=np.inf)
        l_inf_norms.append(l_inf_norm)
        l_inf_relative_norm = l_inf_norm / grad_range * 100
        l_inf_relative_norms.append(l_inf_relative_norm)
        mse = np.mean(np.array(errors) ** 2)
        MSE_vals.append(mse)

        if (l_inf_relative_norm >= details.inf_rel_norm_stanard_lower) \
            and (l_inf_relative_norm <= details.inf_rel_norm_stanard_upper) \
            and (not made_typical_plots):

            plt.plot(details.alpha_values, fit_values)
            plt.plot(details.alpha_values, grads)
            plt.legend(["Poly fit", "Gradient"])
            plt.savefig(f"./results/figures/{details.problem_type}/typical_fit.png")
            plt.clf()

            with open(f"./results/data/{details.problem_type}/typical_fit.pickle", 'wb') as handle:
                pickle.dump(fit_values, handle)

            zero_deriv = get_derivative(grads, details.poly_degree + 1)
            plt.plot(details.alpha_values, grads)
            plt.plot(details.alpha_values, zero_deriv)
            plt.legend(["Gradient", "Zero derivative"])
            plt.savefig(f"./results/figures/{details.problem_type}/typical_deriv.png")
            plt.clf()

            with open(f"./results/data/{details.problem_type}/typical_grads.pickle", 'wb') as handle:
                pickle.dump(grads, handle)

            with open(f"./results/data/{details.problem_type}/typical_deriv.pickle", 'wb') as handle:
                pickle.dump(zero_deriv, handle)

            made_typical_plots = True
        
        if (l_inf_relative_norm >= details.inf_rel_norm_outlier_lower) \
            and (l_inf_relative_norm <= details.inf_rel_norm_outlier_upper) \
            and (not made_outlier_plots):

            plt.plot(details.alpha_values, fit_values)
            plt.plot(details.alpha_values, grads)
            plt.legend(["Poly fit", "Gradient"])
            plt.savefig(f"./results/figures/{details.problem_type}/outlier_fit.png")
            plt.clf()

            with open(f"./results/data/{details.problem_type}/outlier_fit.pickle", 'wb') as handle:
                pickle.dump(fit_values, handle)

            zero_deriv = get_derivative(grads, details.poly_degree + 1)
            plt.plot(details.alpha_values, grads)
            plt.plot(details.alpha_values, zero_deriv)
            plt.legend(["Gradient", "Zero derivative"])
            plt.savefig(f"./results/figures/{details.problem_type}/outlier_deriv.png")
            plt.clf()

            with open(f"./results/data/{details.problem_type}/outlier_grads.pickle", 'wb') as handle:
                pickle.dump(grads, handle)

            with open(f"./results/data/{details.problem_type}/outlier_deriv.pickle", 'wb') as handle:
                pickle.dump(zero_deriv, handle)

            made_outlier_plots = True

    print(f'l_1: mean: {np.mean(l_one_norms)}, median: {np.median(l_one_norms)}, max: {np.max(l_one_norms)}, variance: {np.var(l_one_norms)}')
    print(f'l_infinity: mean: {np.mean(l_inf_norms)}, median: {np.median(l_inf_norms)}, max: {np.max(l_inf_norms)}, variance: {np.var(l_inf_norms)}')
    print(f'relative l_infinity: mean: {np.mean(l_inf_relative_norms)}, median: {np.median(l_inf_relative_norms)}, max: {np.max(l_inf_relative_norms)}, variance: {np.var(l_inf_relative_norms)}')
    print(f'MSE: mean: {np.mean(MSE_vals)}, median: {np.median(MSE_vals)}, max: {np.max(MSE_vals)}, variance: {np.var(MSE_vals)}')

    with open(f"./results/data/{details.problem_type}/l_one_norms.pickle", 'wb') as handle:
        pickle.dump(l_one_norms, handle)
    with open(f"./results/data/{details.problem_type}/l_inf_norms.pickle", 'wb') as handle:
        pickle.dump(l_inf_norms, handle)
    with open(f"./results/data/{details.problem_type}/l_inf_relative_norms.pickle", 'wb') as handle:
        pickle.dump(l_inf_relative_norms, handle)
    with open(f"./results/data/{details.problem_type}/mse_vals.pickle", 'wb') as handle:
        pickle.dump(MSE_vals, handle)

def generate_histograms(details: ExperimentDetails):
    file = open(f"./results/data/{details.problem_type}/l_one_norms.pickle",'rb')
    l_one_norms = pickle.load(file)

    file = open(f"./results/data/{details.problem_type}/l_inf_norms.pickle",'rb')
    l_inf_norms = pickle.load(file)

    file = open(f"./results/data/{details.problem_type}/l_inf_relative_norms.pickle",'rb')
    l_inf_relative_norms = pickle.load(file)

    file = open(f"./results/data/{details.problem_type}/mse_vals.pickle",'rb')
    MSE_vals = pickle.load(file)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(6, 8))
        
    ax1.hist(MSE_vals, bins=30)
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Frequency')
    ax1.set_title("MSE")

    ax2.hist(l_one_norms, bins=30)
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('Frequency')
    ax2.set_title("L1 norm")

    ax3.hist(l_inf_norms, bins=30)
    ax3.set_xlabel('Magnitude')
    ax3.set_ylabel('Frequency')
    ax3.set_title("L-infinity norm")

    ax4.hist(l_inf_relative_norms, bins=30)
    ax4.set_xlabel("% of CaVE gradient range")
    ax4.set_ylabel('Frequency')
    ax4.set_title("Relative L-inf norm")

    fig.suptitle("Goodness of cubic fit on CaVE gradient", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"./results/figures/{details.problem_type}/error_histograms.png")