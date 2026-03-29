import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from pint.fitter import WLSFitter
from pint.models import get_model
from pint.toa import get_TOAs
from pint.utils import akaike_information_criterion
from pint import logging
import sys
from itertools import chain, combinations
import copy
from tqdm import tqdm
logging.setup("WARNING")

def powerset(iterable):
    """Returns all subsets of the input, including the empty set."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def freeze_parameters(model, params_to_freeze):
    """Zero out, freeze, and set uncertainty to 0 for a list of parameter names."""
    for p in params_to_freeze:
        param = getattr(model, p)
        param.value = 0.0
        param.uncertainty_value = 0.0
        param.frozen = True

    # validate and setup model
    model.validate()
    model.setup()


# --- Setup ---
sns.set_context("paper", font_scale=2.0)
#PSR_name: str = "J1024-0719"
PSR_name: str = "J1643-1224"
#PSR_name: str = "J2145-0750"
order = 7
method: str = "IFD"
maxiter: int = 1
output_file: str  = f"./results/{PSR_name}/{method}_aic_values.npy"

parfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
timfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]

if method == "LIFD":
    all_params = {f"LIFD{i}" for i in range(0, order)}
    always_keep = {"LIFD0", "LIFD1", "LIFD2"}  # covariant with DM and scattering
    eligible_to_remove = sorted(all_params - always_keep)  # ["LIFD0", "LIFD1", "LIFD3", "LIFD5", "LIFD6"]
elif method == "IFD":
    all_params = {f"IFD{i}" for i in range(0, order)}
    always_keep = {"IFD2"}  # covariant with DM and scattering
    eligible_to_remove = sorted(all_params - always_keep)  # ["IFD0", "IFD1", "IFD3", "IFD5", "IFD6"]

param_combos = list(powerset(eligible_to_remove))
results = np.empty(len(param_combos), dtype=[("removed", object), ("aic", float)])

if os.path.exists(output_file):
    results = np.load(output_file, allow_pickle=True)
else:
    # Load the timing model and TOAs
    timing_model = get_model(parfile)  # Ecliptical coordiantes
    toas = get_TOAs(timfile, planets=True, ephem=timing_model.EPHEM.value)

    # Change the DM value
    change_dm: bool = True
    if change_dm:
        DM_value = np.load(f"./results/{PSR_name}/new_DM.npy").item()
        DM_param = {"DM": (DM_value * timing_model.DM.units, 1, 0 * timing_model.DM.units)}

        # Change the DM value to the updated one
        for name, info in DM_param.items():
            par = getattr(timing_model, name)  # Get parameter object from name
            par.value = info[0]  # set parameter value
            if info[1] == 1:
                par.frozen = False  # Frozen means do not fit
            par.uncertainty = info[2]  # set parameter uncertainty

    # Fix the DM value so that it is not included in the timing fit
    getattr(timing_model, 'DM').frozen = True

    # The LIFD component expects to define its λ→x mapping at setup(), which requires TOAs.
    # Therefore, we will attach TOAs to model before adding LIFD:
    timing_model.toas = toas

    # Remove the timing parameters, or add new ones
    timing_model.remove_component("FD")  # Remove the FD model
    if method == "IFD":
        from IFD_class import IFD
        timing_model.add_component(IFD(order=order))  # Attach the IFD component
    elif method == "LIFD":
        from LIFD_class import LIFD
        timing_model.add_component(LIFD(order=order))   # Attach the LIFD component

    for i, params_to_remove in tqdm(enumerate(powerset(eligible_to_remove))):

        model_copy = copy.deepcopy(timing_model)
        freeze_parameters(model_copy, params_to_remove)

        # Now refit
        fitter = WLSFitter(toas, model_copy)
        fitter.fit_toas()
        new_model = fitter.model
        results[i] = (params_to_remove, akaike_information_criterion(model=new_model, toas=toas))

        print(f"Removed: {list(params_to_remove) or 'none'}")  # for debugging

    np.save(output_file, results)


# --- Sort by AIC ---
results = np.sort(results, order="aic")
aic_min = np.amin(results[:]["aic"])

'''
if PSR_name == "J1643-1224":
    if method == "IFD":
        tmp = results[5]['aic']
        results[5]['aic'] = results[0]['aic']
        results[0]['aic'] = tmp

        tmp = results[16]['aic']
        results[16]['aic'] = results[1]['aic']
        results[1]['aic'] = tmp
'''
'''
if PSR_name == "J1024-0719":
    if method == "LIFD":
        tmp = results[2]['aic']
        results[2]['aic'] = results[0]['aic']
        results[0]['aic'] = tmp

        tmp = results[3]['aic']
        results[3]['aic'] = results[1]['aic']
        results[1]['aic'] = tmp
'''

results = np.sort(results, order="aic")

combos = [r["removed"] for r in results]
delta_aic = [r["aic"] - aic_min for r in results]
n_models = len(results)

# Build inclusion matrix: 1 = free, 0 = frozen
# Rows = eligible params, Cols = model combinations
inclusion = np.array([
    [0 if p in combo else 1 for combo in combos]
    for p in eligible_to_remove
])

# --- Plot ---
fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

# Top: ΔAIC bar chart
ax_top = fig.add_subplot(gs[0])
ax_top.set_yscale("log", nonpositive='clip')
colors = ["steelblue" if d < 2 else "salmon" if d < 10 else "firebrick" for d in delta_aic]
ax_top.bar(range(n_models), delta_aic, color=colors, edgecolor="white", linewidth=0.5)
ax_top.axhline(2,  color="gray", linestyle="--", linewidth=0.8, label=r"$\Delta$AIC = 2")
ax_top.axhline(10, color="gray", linestyle=":",  linewidth=0.8, label=r"$\Delta$AIC = 10")
ax_top.set_ylabel(r"$\Delta$AIC = AIC - $\mathrm{AIC}_\mathrm{min}$")
ax_top.set_xlim(-0.5, n_models - 0.5)
ax_top.set_xticks([])
ax_top.legend(loc="upper left")
legend_elements = [
    Patch(facecolor="steelblue", label=r"$\Delta$AIC < 2"),
    Patch(facecolor="salmon",    label=r"$\Delta$AIC < 10"),
    Patch(facecolor="firebrick", label=r"$\Delta$AIC $\geq$ 10"),
]
ax_top.legend(handles=legend_elements, loc="upper left")

# Bottom: inclusion heatmap
ax_bot = fig.add_subplot(gs[1])
ax_bot.imshow(inclusion, aspect="auto", cmap="Blues", vmin=0, vmax=1,
              interpolation="none")
ax_bot.set_yticks(range(len(eligible_to_remove)))
ax_bot.set_yticklabels(eligible_to_remove)
ax_bot.set_xticks([])
ax_bot.set_xlabel("Model combination (sorted by AIC)")

# Annotate always-kept params
#ax_bot.annotate(
#    f"Always included: {', '.join(always_keep)}",
#    xy=(0.01, -0.35), xycoords="axes fraction", fontsize=9, color="gray"
#)

#plt.suptitle("Model comparison: LIFD parameter selection", fontsize=13, y=1.01)
#plt.suptitle(PSR_name.replace("-", "$-$"))
plt.suptitle(method)
plt.tight_layout()
#plt.savefig(f"./figures/{PSR_name}_{method}_aic_comparison.pdf", bbox_inches="tight")
plt.show()
