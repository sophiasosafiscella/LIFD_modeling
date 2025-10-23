import numpy as np
import pypulse as pyp
from glob import glob
from tqdm import tqdm
import math

# Constants
INVERSE_PHI = (math.sqrt(5) - 1) / 2  # Golden ratio inverse: 1 / phi
DEFAULT_TOLERANCE = 1e-4
DM_SEARCH_RANGE_PERCENT = 10.0  # Percentage of nominal DM to search


def calculate_snr(files, dm_value, template_profile):
    """Calculate the average S/N for a given DM value"""
    snr_values = np.zeros(len(files))
    for j, file in tqdm(enumerate(files)):
        ar = pyp.Archive(file, prepare=False, verbose=False)
        ar.dedisperse(DM=dm_value, wcfreq=True)
        ar.pscrunch()
        ar.center()
        ar.fscrunch()
        ar.tscrunch()
        snr_values[j] = ar.fitPulses(template_profile, nums=[5])[0]
    average_snr = np.mean(snr_values)
    return average_snr


def golden_section_search(snr_func, lower_bound, upper_bound, tolerance=DEFAULT_TOLERANCE):
    """
    Perform golden section search to find the maximum of a unimodal function.

    Args:
        function: Function to maximize (should take single argument)
        lower_bound: Lower bound of search interval
        upper_bound: Upper bound of search interval
        tolerance: Convergence tolerance for the search interval width

    Returns:
        tuple: (optimal_value, max_objective_value, num_iterations)
    """
    a = lower_bound
    b = upper_bound
    iteration = 1

    while b - a > tolerance:
        print(f"Iteration {iteration}. Searching in the interval [{a:.4f}, {b:.4f}]")

        # Calculate interior points using golden ratio
        c = b - (b - a) * INVERSE_PHI  # Point closer to b
        d = a + (b - a) * INVERSE_PHI  # Point closer to a

        # Evaluate function at interior points
        if snr_func(c) > snr_func(d):
            b = d
        else:
            a = c

        iteration += 1

    optimal_dm = (a + b) / 2
    optimal_snr = snr_func(optimal_dm)

    return optimal_dm, optimal_snr, iteration - 1


def load_template_profile(psr_name, template_dir):
    """Load the template profile for a given pulsar."""
    template_file = glob(f"{template_dir}{psr_name}.*.GUPPI.15y.x.sum.sm")[0]
    return pyp.Archive(template_file).getSinglePulses()


def main():
    """Main execution function for DM optimization using golden section search."""
    # Set up parameters and files
    psr_name = "J2145-0750"
    output_file = f"./{psr_name}/{psr_name}_DM_results_2.csv"
    ar_files = glob(f"./{psr_name}/ff_files/*.ff")

    # Load the template
    template_dir = "/home/svsosafiscella/PycharmProjects/NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/template/"
    template_profile = load_template_profile(psr_name, template_dir)

    # Calculate the nominal DM and search range
    DM_0 = pyp.Archive(ar_files[0], prepare=True).getDM()
    print(f"Nominal DM = {DM_0:.4f}")

    delta_dm = DM_0 / 100.0 * DM_SEARCH_RANGE_PERCENT
    lower_bound = DM_0 - delta_dm
    upper_bound = DM_0 + delta_dm

    # Create objective function for golden section search
    def snr_func(dm_value):
        return calculate_snr(ar_files, dm_value, template_profile)

    # Perform golden section search. This method is only valid for a function with a single maximum.
    final_dm, final_snr, num_iterations = golden_section_search(snr_func, lower_bound, upper_bound)

    print(f"DM = {final_dm}")
    print(f"SNR = {final_snr}")


if __name__ == "__main__":
    main()