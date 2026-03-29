import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

#PSR_name: str = "B1937+21"
#PSR_name: str = "J1012+5307"
#PSR_name: str = "J1022+1001"
#PSR_name: str = "J1024-0719"
PSR_name: str = "J1643-1224"
#PSR_name: str = "J2145-0750"
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.5})
fig, ax = plt.subplots(nrows=1, ncols=1)

# FD
FD_df = pd.read_pickle(f"../results/{PSR_name}/FD_DMX.pkl")
FD_DMX = FD_df[FD_df.index.str.startswith('DMX_')]
FD_DMX_errors = [x.value for x in FD_DMX.Error.to_numpy()]

FD_DMXR1 = FD_df[FD_df.index.str.startswith('DMXR1_')].Value.to_numpy()
FD_DMXR2 = FD_df[FD_df.index.str.startswith('DMXR2_')].Value.to_numpy()
FD_DMXR_centers = FD_DMXR1 + (FD_DMXR2 - FD_DMXR1)/2.0

# IFD
IFD_df = pd.read_pickle(f"../results/{PSR_name}/IFD_DMX.pkl")
IFD_DMX = IFD_df[IFD_df.index.str.startswith('DMX_')]
IFD_DMX_errors = [x.value for x in IFD_DMX.Error.to_numpy()]

IFD_DMXR1 = IFD_df[IFD_df.index.str.startswith('DMXR1_')].Value.to_numpy()
IFD_DMXR2 = IFD_df[IFD_df.index.str.startswith('DMXR2_')].Value.to_numpy()
IFD_DMXR_centers = IFD_DMXR1 + (IFD_DMXR2 - IFD_DMXR1)/2.0

# LIFD
LIFD_df = pd.read_pickle(f"../results/{PSR_name}/LIFD_DMX.pkl")
LIFD_DMX = LIFD_df[LIFD_df.index.str.startswith('DMX_')]
LIFD_DMX_errors = [x.value for x in LIFD_DMX.Error.to_numpy()]

LIFD_DMXR1 = LIFD_df[LIFD_df.index.str.startswith('DMXR1_')].Value.to_numpy()
LIFD_DMXR2 = LIFD_df[LIFD_df.index.str.startswith('DMXR2_')].Value.to_numpy()
LIFD_DMXR_centers = LIFD_DMXR1 + (LIFD_DMXR2 - LIFD_DMXR1)/2.0

ax.errorbar(x=FD_DMXR_centers, y=FD_DMX.Value, yerr=FD_DMX_errors, fmt='o', c='C0', label='FD', alpha=0.5)
ax.errorbar(x=IFD_DMXR_centers, y=IFD_DMX.Value, yerr=IFD_DMX_errors, fmt='o', c='C1', label='IFD', alpha=0.6)
ax.errorbar(x=LIFD_DMXR_centers, y=LIFD_DMX.Value, yerr=LIFD_DMX_errors, fmt='o', c='C2', label='LIFD', alpha=0.8)


ax.set_xlabel("DMX Window Center [MDJ]")
ax.set_ylabel("DMX [$\mathrm{pm}/\mathrm{cm^3}$]")
plt.legend()
plt.tight_layout()
plt.savefig(f"./{PSR_name}_DMX.pdf")
plt.show()
sys.exit()
