import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples=[]
colors=[]
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 
import os    
from scipy import misc
for img in os.listdir('c:/Users/User/workspace/DAT210x/Module4/Datasets/ALOI/32/'):
    samples.append(misc.imread('c:/Users/User/workspace/DAT210x/Module4/Datasets/ALOI/32/'+ img).reshape(-1))
for img_i in os.listdir('c:/Users/User/workspace/DAT210x/Module4/Datasets/ALOI/32i/'):
    samples.append(misc.imread('c:/Users/User/workspace/DAT210x/Module4/Datasets/ALOI/32i/'+ img_i ).reshape(-1))
for img in samples:
    colors.append('b')
for img_i in samples:
    colors.append('r')
    
df=pd.DataFrame(samples)
from sklearn import manifold
iso = manifold.Isomap(n_neighbors=6, n_components=3)
iso.fit(df)
manifold = iso.transform(df)

manifold_df=pd.DataFrame(manifold)

manifold_df.plot.scatter(x=0, y=1)
manifold_df.plot.scatter()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('component 0')
ax.set_ylabel('component 1')
ax.set_zlabel('component 2')
ax.scatter(manifold_df.iloc[:,0], manifold_df.iloc[:,1], manifold_df.iloc[:,2], c=colors, marker='.')
plt.show()

# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 


#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 



#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 



#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 




#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 



plt.show()

