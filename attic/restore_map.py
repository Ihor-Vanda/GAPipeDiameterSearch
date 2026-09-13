import pandas as pd
import wntr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings

INP_FILE = "InputData/Balerma/Balerma.inp"
SOLUTION_FILE = "OutputData/tables/solution_champion.csv"
OUTPUT_PLOT = "OutputData/plots/regenerated_map.png"

def main():
    print("Regenerating map from CSV...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wn = wntr.network.WaterNetworkModel(INP_FILE)

    df = pd.read_csv(SOLUTION_FILE)
    diam_col = [c for c in df.columns if "Diameter" in c][0]
    
    pipe_diams = dict(zip(df["Pipe ID"], df[diam_col]))
    
    attr_series = pd.Series(pipe_diams)
    
    plt.figure(figsize=(12, 10))
    wntr.graphics.plot_network(wn, link_attribute=attr_series, node_size=15, 
                               link_width=2, link_cmap='viridis', 
                               title="Optimized Pipe Diameters (Regenerated)")
    
    unique_diams = sorted(df[diam_col].unique())
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min(unique_diams), vmax=max(unique_diams))
    
    patches = []
    for d in unique_diams:
        color = cmap(norm(d))
        patches.append(mpatches.Patch(color=color, label=f"{d}"))
        
    plt.legend(handles=patches, title=f"Diameters", 
               bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Done! Map saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()