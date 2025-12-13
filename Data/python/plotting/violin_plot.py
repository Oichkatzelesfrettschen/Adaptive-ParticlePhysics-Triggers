import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hdf5plugin
from matplotlib.lines import Line2D
import mplhep as hep

hep.style.use("CMS")

# --- CMS header, with configurable positions ---
def add_cms_header(fig, left_x=0.20, right_x=0.94, y=0.98):
    # left label
    fig.text(
        left_x, y, "CMS Open Data",
        ha="left", va="top",
        fontweight="bold", fontsize=24
    )
    # right label
    fig.text(
        right_x, y, "Run 283408",
        ha="right", va="top",
        fontsize=24
    )

# ======= Calculate HT manually =======
def load_and_calculate_ht(filename):
    with h5py.File(filename, "r") as f:
        n_events = f["j0Eta"].shape[0]
        n_jets = 8
        ht_values = np.zeros(n_events)

        for i in range(n_jets):
            eta = f[f"j{i}Eta"][:]
            pt = f[f"j{i}Pt"][:]
            mask = (pt > 20) & (np.abs(eta) < 2.5)
            ht_values += pt * mask

    return ht_values

# ======= data =======
file_name = "new_Data/data_Run_2016_283408_longest.h5"
Ht_manual = load_and_calculate_ht(file_name)
Ht_manual = Ht_manual[Ht_manual > 0]

# ======= Parameters =======
chunk_size = 20000
num_chunks = len(Ht_manual) // chunk_size
time_fraction = np.linspace(0, 1, len(Ht_manual), endpoint=False)

# ======= DataFrame for violin =======
df = pd.DataFrame({
    "HT": Ht_manual,
    "time_frac": time_fraction
})

n_time_bins = 10
df["time_bin"] = pd.cut(df["time_frac"], bins=n_time_bins)

# ======= Colormap =======
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=0, vmax=1)

ht_min, ht_max = 25, 300
num_bins_ht = 20
ht_bins = np.linspace(ht_min, ht_max, num_bins_ht + 1)

# ================= FIGURE (a) Histogram =================
fig, ax1 = plt.subplots(figsize=(7, 6))

for i in range(n_time_bins):
    start, end = int(i*len(Ht_manual)/n_time_bins), int((i+1)*len(Ht_manual)/n_time_bins)
    chunk = Ht_manual[start:end]
    frac = i / n_time_bins
    color = cmap(norm(frac))

    ax1.hist(chunk, bins=ht_bins, histtype="step",
             color=color, linewidth=1.5)

ax1.set_xlabel("HT", loc='center')
ax1.set_ylabel("Number of Events", loc='center')
ax1.set_yscale('log')
ax1.set_xlim(ht_min, ht_max)
ax1.grid(False)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar1 = fig.colorbar(sm, ax=ax1)
cbar1.set_label("Time (Fraction of Run)", loc='center')

# Leave some room on top for the header
#plt.subplots_adjust(top=0.90)
fig.tight_layout()

add_cms_header(fig, left_x=0.18, right_x=0.83, y=0.98)

fig.savefig("Figure2_a.pdf", dpi=300)
plt.close(fig)

# ================= FIGURE (b) Violin =================
fig, ax2 = plt.subplots(figsize=(7, 6))
ht_min, ht_max = 20, 175

sns.violinplot(
    x="time_bin", y="HT", data=df,
    inner=None, cut=0,
    bw=0.4,
    palette="viridis", ax=ax2
)

median_color = "#4e8b50"
quartile_color = "#d98c4c"

for i, (bin_label, group) in enumerate(df.groupby("time_bin")):
    q1 = group["HT"].quantile(0.25)
    q2 = group["HT"].quantile(0.50)
    q3 = group["HT"].quantile(0.75)
    ax2.hlines([q1], i-0.25, i+0.25, color=quartile_color, linewidth=2)
    ax2.hlines([q2], i-0.25, i+0.25, color=median_color, linewidth=2)
    ax2.hlines([q3], i-0.25, i+0.25, color=quartile_color, linewidth=2)

legend_elements = [
    Line2D([0], [0], color=quartile_color, lw=2, label="Q1 / Q3"),
    Line2D([0], [0], color=median_color, lw=2, label="Median")
]
ax2.legend(handles=legend_elements, title="Quartiles", loc="upper right", frameon=True)

ax2.set_xlabel("Time (Fraction of Run)", loc='center')
ax2.set_ylabel("HT", loc='center')
ax2.set_ylim(ht_min, ht_max)

ax2.set_xticks(np.arange(n_time_bins))
ax2.set_xticklabels(
    [f"{b.right:.1f}" for b in df["time_bin"].cat.categories]
)

ax2.tick_params(axis="both", which="major")
ax2.grid(False)

# Per il violin lo sposto un po' a sinistra e un filo più in basso
#plt.subplots_adjust(top=0.90)
fig.tight_layout()

add_cms_header(fig, left_x=0.18, right_x=0.94, y=0.98)

fig.savefig("Figure2_b.pdf", dpi=300)
plt.close(fig)
