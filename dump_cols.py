import pandas as pd
df50 = pd.read_csv(r'd:\LSC\pipeline_output\lsc50_interim.csv')
df54 = pd.read_csv(r'd:\LSC\pipeline_output\lsc54_interim.csv')
df70 = pd.read_csv(r'd:\LSC\pipeline_output\lsc70_interim.csv')
with open(r'd:\LSC\cols.txt', 'w') as f:
    f.write("LSC50:\n" + ",".join(df50.columns) + "\n\n")
    f.write("LSC54:\n" + ",".join(df54.columns) + "\n\n")
    f.write("LSC70:\n" + ",".join(df70.columns) + "\n\n")
