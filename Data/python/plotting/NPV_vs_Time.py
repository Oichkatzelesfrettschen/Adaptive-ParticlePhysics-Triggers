import hdf5plugin
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# --- CMS-like style ---
hep.style.use("CMS")

# --- CMS header on the figure ---
def add_cms_header(fig):
    # left
    fig.text(
        0.12, 0.98, "CMS Open Data",
        ha="left", va="top",
        fontweight="bold", fontsize=24
    )
    # right
    fig.text(
        0.97, 0.98, "Run 283408",
        ha="right", va="top", fontsize=24
    )

# --- Input file ---
file_name = "new_Data/data_Run_2016_283408_longest.h5"

with h5py.File(file_name, "r") as h5file:
    npv_data = h5file["PV_npvsGood"][:]

npv_data = npv_data[npv_data > 0]

# --- Time-binning parameters ---
chunk_size = 100000
num_chunks = len(npv_data) // chunk_size
time_fraction = np.linspace(0, 1, num_chunks, endpoint=True)

# --- Mean and statistical uncertainty ---
avg_npv, std_npv = [], []
for i in range(num_chunks):
    start, end = i * chunk_size, (i + 1) * chunk_size
    chunk = npv_data[start:end]
    avg_npv.append(np.mean(chunk))
    std_npv.append(np.std(chunk))

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 6))

ax.errorbar(
    time_fraction,
    avg_npv,
    yerr=std_npv,
    fmt='o',
    color='royalblue',
    markersize=7,
    markeredgecolor='black',
    capsize=1,
    elinewidth=1,
    label="Average Num Primary Vertices per chunk"
)

# --- Labels and style ---
ax.set_xlabel("Time (Fraction of Run)", labelpad=8, loc='center')
ax.set_ylabel("Average Num Primary Vertices", labelpad=8, loc='center')

ax.tick_params(axis='both', direction='in', top=True, right=True)
ax.grid(alpha=0.3, linestyle='--')

from matplotlib.ticker import MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

ax.legend(
    frameon=True,
    loc='upper right',
    fontsize=18
)

# --- Add CMS header and adjust layout ---
add_cms_header(fig)
plt.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.15)

fig.savefig("Figure1.pdf", dpi=300)
plt.show()
