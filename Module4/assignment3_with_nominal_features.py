import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper

# Look pretty...
matplotlib.style.use('ggplot')


# Do * NOT * alter this line, until instructed!
scaleFeatures = True


# TODO: Load up the dataset and rem pro at this
# by now ;-)
#
# .. your code here
original_df = pd.read_csv ('Datasets/kidney_disease.csv')
# Rows that have a nan. You should be a_disease.csv')
new_df = original_df.dropna()     


# Create some color coded labels; the actual label feature
# will be removed prior to executing PCA, since it's unsupervised.
# You're only labeling by color so you can see the effects of PCA
labels = ['red' if i=='ckd' else 'green' for i in new_df.classification]


# TODO: Instead of using an indexer to select just the bgr, rc, and wc, 
#Alter your code so that you only drop the id and classification columns.
#Be sure you select the right axis for columns and not rows, otherwise Pandas will complain!
#
# .. your code here ..
new_df.dtypes
df=new_df.drop(['id', 'classification'],1)

# TODO: Print out and check your dataframe's dtypes. You'll probably
# want to call 'exit()' after you print it out so you can stop the
# program's execution.
df.dtypes

# Does everything look like it should / properly numeric? 
#If not, make code changes to coerce the remaining column(s).
#
df.wc=pd.to_numeric(df.wc)
df.rc=pd.to_numeric(df.rc)
df.pcv=pd.to_numeric(df.pcv)
# For the remaining 10 nominal features, properly encode them by as explained 
# the Feature Representation section by creating new, boolean columns using
# Pandas .get_dummies(). You should be able to carry that out with a single
# line of code. Run your assignment again and see if your results have changed at all.
# .. your code here ..
df = pd.get_dummies(df,columns=['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])


# TODO: PCA Operates based on variance. The variable with the greatest
# variance will dominate. Go ahead and peek into your data using a
# command that will check the variance of every feature in your dataset.
# Print out the results. Also print out the results of running .describe
# on your dataset.
df.describe()

# TODO: This method assumes your dataframe is called df. If it isn't,
# make the appropriate changes. Don't alter the code in scaleFeatures()
# just yet though!
#
# .. your code adjustment here ..
if scaleFeatures: df = helper.scaleFeatures(df)

# TODO: Run PCA on your dataset and reduce it to 2 components
# Ensure your PCA instance is saved in a variable called 'pca',
# and that the results of your transformation are saved in 'T'.
#
# .. your code here ..
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df)
T = pca.transform(df)

# Plot the transformed data as a scatter plot. Recall that transforming
# the data will result in a NumPy NDArray. You can either use MatPlotLib
# to graph it directly, or you can convert it to DataFrame and have pandas
# do it for you.

# Since we've already demonstrated how to plot directly with MatPlotLib in
# Module4/assignment1.py, this time we'll convert to a Pandas Dataframe.
#
# Since we transformed via PCA, we no longer have column names. We know we
# are in P.C. space, so we'll just define the coordinates accordingly:
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()



