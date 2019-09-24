import pandas as pd

df = pd.DataFrame(columns=['s', 'val'], data=[[1, True], [1, True], [1, False], [2, False], [2, False]])
mm = df.groupby('s').max()
m = df.groupby('s').min()

print(mm[mm['val'] == True].index)