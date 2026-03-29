import numpy as np
import matplotlib.pyplot as plt
from pint.models import get_model
from pint.toa import get_TOAs
from glob import glob
import seaborn as sns
import sys

sns.set_context(font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_style("ticks")

psr_list = ["B1937+21", "J0610-2100", "J1012+5307", "J1022+1001", "J1024-0719", "J1643-1224", "J1713+0747", "J1909-3744", "J1918-0642", "J2145-0750", "J2302+4442"]
old_DM_list = np.zeros(len(psr_list))
new_DM_list = [round(np.load(f"../results/{psr}/new_DM.npy").item(), 4) for psr in psr_list]
n_TOAs_list = np.zeros(len(psr_list))

for i, PSR_name in enumerate(psr_list):
    parfile: str = glob(f"../NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
    timfile: str = glob(f"../NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]
    tm = get_model(parfile)
    toas = get_TOAs(timfile, planets=True, ephem=tm.EPHEM.value)
    old_DM_list[i] = round(tm.DM.value, 4)
    n_TOAs_list[i] = toas.ntoas

diff_arr = np.round(np.divide(new_DM_list - old_DM_list, old_DM_list) * 100.0, 4)

print(old_DM_list)
print(new_DM_list)
print(n_TOAs_list)


with np.printoptions(suppress=True):
    print(diff_arr)

sys.exit()

fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.scatter(psr_list, old_DM_list, label="Old DM", marker="*", color="C0")
ax.scatter(psr_list, new_DM_list, label="New DM", marker="o", color="C1")
ax.legend()
plt.show()
