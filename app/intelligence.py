import utils
import pandas as pd

def compute_global_risk(df):
    results = []

    for _, row in df.iterrows():
        p, _ = utils.predict_risk(row.to_frame().T)
        results.append({
            "country": row["country"],
            "risk": p
        })

    return pd.DataFrame(results).sort_values("risk", ascending=False)
