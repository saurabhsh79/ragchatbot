import pandas as pd

def highlight_better(df, ticker1, ticker2):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for i, kpi in enumerate(df["KPI"]):
        v1 = df.loc[i, f"{ticker1}"]
        v2 = df.loc[i, f"{ticker2}"]
        try:
            n1 = float(v1 if v1 != "N/A" else "nan")
            n2 = float(v2 if v2 != "N/A" else "nan")
            if "PE" in kpi or "Price" in kpi:
                if n1 < n2:
                    styles.loc[i, f"{ticker1}"] = "background-color: lightgreen"
                    styles.loc[i, f"{ticker2}"] = "background-color: salmon"
                elif n2 < n1:
                    styles.loc[i, f"{ticker2}"] = "background-color: lightgreen"
                    styles.loc[i, f"{ticker1}"] = "background-color: salmon"
                else:
                    styles.loc[i, [f"{ticker1}", f"{ticker2}"]] = "background-color: lightgray"
            else:
                if n1 > n2:
                    styles.loc[i, f"{ticker1}"] = "background-color: lightgreen"
                    styles.loc[i, f"{ticker2}"] = "background-color: salmon"
                elif n2 > n1:
                    styles.loc[i, f"{ticker2}"] = "background-color: lightgreen"
                    styles.loc[i, f"{ticker1}"] = "background-color: salmon"
                else:
                    styles.loc[i, [f"{ticker1}", f"{ticker2}"]] = "background-color: lightgray"
        except:
            styles.loc[i, [f"{ticker1}", f"{ticker2}"]] = "background-color: lightgray"
    return styles
