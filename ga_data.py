import os
import sys
import pandas as pd
from ga_config import CONFIG

def load_config(cost_file, h_min, units, silent=False):
    if not os.path.exists(cost_file):
        if not silent: print(f"[ERROR] Cost file '{cost_file}' not found.")
        sys.exit(1)
    
    try:
        try:
            df = pd.read_csv(cost_file)
            if len(df.columns) < 2: df = pd.read_csv(cost_file, sep=';')
        except:
            df = pd.read_csv(cost_file, sep=None, engine='python')

        df.columns = df.columns.str.strip()
        diam_col = next((c for c in ["Diameter", "Diam", "D", "diameter", "size"] if c in df.columns), None)
        cost_col = next((c for c in ["Cost", "cost", "Price", "price", "UnitCost"] if c in df.columns), None)
        
        if not diam_col or not cost_col: raise KeyError("Columns Diameter/Cost not found")

        df = df.sort_values(by=diam_col)
        CONFIG["diameters_raw"] = df[diam_col].tolist()
        
        if units == "mm":
            CONFIG["diameters_m"] = [d / 1000.0 for d in df[diam_col]]
            if not silent: print("[Config] Units: Millimeters (converted to meters / 1000)")
        else:
            CONFIG["diameters_m"] = [d * 0.0254 for d in df[diam_col]]
            if not silent: print("[Config] Units: Inches (converted to meters * 0.0254)")
            
        CONFIG["costs"] = dict(zip(df[diam_col], df[cost_col]))
        CONFIG["h_min"] = h_min
        CONFIG["unit_system"] = units
        
        if not silent: print(f"[Config] Loaded {len(CONFIG['diameters_raw'])} pipe types.")
            
    except Exception as e:
        if not silent: print(f"[ERROR] Failed to load costs: {e}")
        sys.exit(1)