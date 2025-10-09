import pandas as pd

df = pd.DataFrame([[[1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8]]] * 3, columns=['A', 'B'])

print(df)

def function(x):
    return x[6]-x[5]

df = df.applymap(function)

print(df)