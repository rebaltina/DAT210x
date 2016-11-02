import pandas as pd 

import html5lib

# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
tables = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2')
df = tables[0]
# TODO: Rename the columns so that they match the
# column definitions provided to you on the website
#
df.columns = [ 'RK', 'PLAYER', 'TEAM', 'GP', 'G', 'A', 'PTS', '+/-', 'PIM', 'PTS/G', 'SOG', 'PCT', 'GWG', 'PP_G', 'PP_A', 'SH_G', 'SH_A']
 

# TODO: Get rid of any row that has at least 4 NANs in it
#
df = df.dropna(axis=0, thresh=4)


# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
selected_df = df.drop([1,25,37,13])


  

# TODO: Get rid of the 'RK' column
#
del selected_df['RK']



# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
selected_df = selected_df.reset_index(drop=True) 


# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
#

selected_df.dtypes
for i in range(2,16):
        selected_df.iloc[:,i] = pd.to_numeric(selected_df.iloc[:,i], errors='coerce')

# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.

len(selected_df)
len(selected_df.PCT.unique())
(selected_df.GP[15])+(selected_df.GP[16])