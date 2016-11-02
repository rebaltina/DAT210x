import pandas as pd
import matplotlib.pyplot as plt


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
df = pd.read_csv ('c:/Users/User/workspace/DAT210x/Module3/Datasets/wheat.data')
#
# TODO: Drop the 'id' feature
# 
# .. your code here ..
df_drop2 = df.drop(df.columns[[0]], axis=1)

#
# TODO: Compute the correlation matrix of your dataframe
# 
# .. your code here ..
df_drop2.corr()

#
# TODO: Graph the correlation matrix using imshow or matshow
# 
# .. your code here ..
plt.imshow(df_drop2.corr(), cmap=plt.cm.Blues, interpolation='nearest')

plt.show()


