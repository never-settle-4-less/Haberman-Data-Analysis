#!/usr/bin/env python
# coding: utf-8

# ## Dataset Details
# 
# >**Haberman's Survival Data**<br>
# >Url of dataset :<br>
# https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/<br>
# (or)<br>
# https://www.kaggle.com/gilsousa/habermans-survival-data-set/version/1
# 
# **Relevant Information:**<br>
#    The dataset contains cases from a study that was conducted between
#    1958 and 1970 at the University of Chicago's Billings Hospital on
#    the survival of patients who had undergone surgery for breast
#    cancer.<br> 
#    
# Number of Instances: 306<br>
# 
# Number of Attributes: 4 (including the class attribute)<br>
# 
# **Attribute Information:**<br>
#    1. Age of patient at time of operation (numerical)<br>
#    2. Patient's year of operation (year - 1900, numerical)<br>
#    3. Number of positive axillary nodes detected (numerical)<br>
#    4. Survival status (class attribute)<br>
#          1 = the patient survived 5 years or longer<br>
#          2 = the patient died within 5 year<br>
# 
# Missing Attribute Values: None<br>
# 

# ## Loading Data

# **Loading CSV File**
# 
# CSV Files can be loaded in 3 possible ways:<br>
# 
# 1.With the help of standard python library (from Scratch)<br>
# 2.Using Numpy<br>
# 3.Using Pandas <br>

# **Loading CSV File with the help of Python Standard libary**

# In[1]:


import csv
import numpy
filename = "C:\\Users\\User\\Desktop\\Haberman\\haberman dataset.csv"
raw_data = open(filename,'r')
reader = csv.reader(raw_data,delimiter = ',', quoting = csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')
print(data.shape)


# **Load CSV Files with NumPy**

# In[2]:


from numpy import loadtxt
filename = "C:\\Users\\User\\Desktop\\Haberman\\haberman dataset.csv"
raw_data = open(filename,'r')
data = loadtxt(raw_data,delimiter = ',')
print(data.shape)


# **Loading CSV File directly from the URL**<br>

# In[3]:


from numpy import loadtxt
from urllib.request import urlopen
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter = ',')
print(dataset.shape)


# **Load CSV Files with Pandas**

# In[4]:


from pandas import read_csv
filename = "C:\\Users\\User\\Desktop\\Haberman\\haberman dataset.csv"
data = read_csv(filename)
print(data.tail)


# In[5]:


# Header Row with column names is not counted in the above approach. To deal with it, lets add names:

from pandas import read_csv
filename = "C:\\Users\\User\\Desktop\\Haberman\\haberman dataset.csv"
names = ['Age of patient at time of operation','Patient\'s year of operation','Number of positive axillary nodes detected ','Survival status']
data = read_csv(filename, names = names)
print(data.shape)


# **Loading Data directly from URL Using Pandas**

# In[6]:


from pandas import read_csv
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'
filename = "C:\\Users\\User\\Desktop\\Haberman\\haberman dataset.csv"
names = ['Age of patient at time of operation','Patient\'s year of operation','Number of positive axillary nodes detected ','Survival status']
data = read_csv(filename, names = names)
print(data.head(5))


# ## Descriptive Statistics on Data 

# **Peeking at the raw data**
# 
# There is no substitute for looking at the raw data. Looking at the raw data can reveal insights
# that you cannot get any other way. It can also plant seeds that may later grow into ideas on
# how to better pre-process and handle the data for machine learning tasks. You can review the
# first 30 rows of your data using the head() function on the Pandas DataFrame.
# 

# In[7]:


from pandas import read_csv
filename = "C:\\Users\\User\\Desktop\\Haberman\\haberman dataset.csv"
names = ['Age','Year of Op','No of positive axillary','Survival status']
data = read_csv(filename, names = names)
peek = data.head(30)
print(peek)


# **Looking at Dimensions of the raw data**
# 
# Its very important that we look at the overall dimensions of the data both in  terms of rows and columns. <br>
# 
# For too many rows some algorithms may take too long to train. Too few and perhaps you do not have enough data to train the algorithms.<br>
# 
# For too many features/colums some algorithms can be distracted or suffer poor performance due to the curse of dimensionality.<br>

# In[8]:


print(data.shape)


# **Data Type for Each Attribute**
# 
# The type of each attribute is important. Strings may need to be converted to 
# floating point
# values or integers to represent categorical or ordinal values. You can get an idea of the types of
# attributes by peeking at the raw data, as above. You can also list the data types used by the
# DataFrame to characterize each attribute using the dtypes property.

# In[9]:


types = data.dtypes
print(types)


# **Descriptive Statistics**
# 
# Descriptive statistics can give you great insight into the shape of each attribute. Often you can
# create more summaries than you have time to review. The describe() function on the Pandas
# DataFrame lists 9 statistical properties of each attribute. They are:<br>
# 
#  **Count.<br>
#  Mean.<br>
#  Standard Deviation.<br>
#  Minimum Value.<br>
#  25th Percentile.<br>
#  50th Percentile (Median).<br>
#  75th Percentile.<br>
#  Maximum Value.<br>
#  Minimum Value.<br>**

# In[10]:


from pandas import set_option
set_option('display.width',200)
set_option('precision',3)
print(data.describe())


# **Analyzing the Class Distributions (for classification problems only)**
# 
# On classification problems you need to know how balanced the class values are. Highly imbalanced
# problems (a lot more observations for one class than another) are common and may need special
# handling in the data preparation stage of the project

# In[11]:


from pandas import read_csv
filename = "C:\\Users\\User\\Desktop\\Haberman\\haberman dataset.csv"
names = ['age','year','axillary_nodes','class']
newdata = read_csv(filename, names = names)
class_counts = newdata.groupby('class').size()
print(class_counts)


# **Correlations between Attributes**
# 
# Correlation refers to the relationship between two variables and how they may or may not
# change together. The most common method for calculating correlation is Pearson's Correlation
# Coefficient, that assumes a normal distribution of the attributes involved. **A correlation of -1
# or 1 shows a full negative or positive correlation respectively. Whereas a value of 0 shows no
# correlation at all.**<br>
# 
# Some machine learning algorithms like linear and logistic regression can suffer
# poor performance if there are highly correlated attributes in your dataset. As such, it is a good
# idea to review all of the pairwise correlations of the attributes in your dataset. You can use the
# corr() function on the Pandas DataFrame to calculate a correlation matrix.

# **Pair-wise Pearson correlations**

# In[12]:


from pandas import set_option
set_option('display.width',200)
set_option('precision',3)
correlations = newdata.corr(method = 'pearson')
print(correlations)


# **Skew of Univariate Distributions**
# 
# *Skew refers to a distribution that is assumed Gaussian (normal or bell curve) that is shifted or
# squashed in one direction or another .*<br>
# 
# Many **machine learning algorithms assume a Gaussian
# distribution**. Knowing that an attribute has a skew may allow you to perform data preparation
# to correct the skew and later improve the accuracy of your models.
# 
# You can calculate the skew
# of each attribute using the **skew()** function on the Pandas DataFrame

# In[13]:


skew = newdata.skew()
print(skew)


# ## Visualization of Data 
# (for better understanding)

# newdata['age'].value_counts().plot.bar(title = 'Frequency Distibution Of Ages')

# **Univariate Analysis**

# **Frequency Distribution Tables**

# In[342]:


import pandas as pd
names = ['age','year_of_op','axillary_nodes','survival_status']
haberman = pd.read_csv("C:\\Users\\User\\Desktop\\Haberman\\haberman dataset.csv" , names = names)


# In[343]:


haberman.head(5)


# In[344]:


haberman.tail(5)


# In[279]:


print(haberman.shape)


# In[280]:


haberman.describe()


# **Frequency Distribution Table for survival_status Column**

# In[281]:


#Method 1 

haberman['survival_status'].value_counts()


# In[282]:


#Method 2

v = haberman.groupby('survival_status').size().reset_index(name = 'count')
freqtableofclasses = pd.DataFrame(v)
freqtableofclasses


# **Frequency Distribution Table for Age Column**

# In[283]:


f = haberman.groupby('age').size().reset_index(name = 'count')
freqtableofage = pd.DataFrame(f)
freqtableofage


# **Frequency Distribution Table for Years_of_Operation Column**

# In[284]:


m = haberman.groupby('year_of_op').size().reset_index(name = 'count')
freqtableofyear = pd.DataFrame(m)
freqtableofyear


# **Frequency Distribution Table for Axillary_nodes Column**

# In[285]:


b = haberman.groupby('axillary_nodes').size().reset_index(name = 'count')
freqtableofaxils = pd.DataFrame(b)
freqtableofaxils


# **Frequency Distribution Table for Survival Status and Age**

# In[286]:


o = haberman.groupby(['survival_status','age']).size().reset_index(name = 'count')
freqtableofclassandage = pd.DataFrame(o)
freqtableofclassandage 


# **Frequency Distribution Table for Survival Status and Year Of Operation**

# In[287]:


p = haberman.groupby(['survival_status','year_of_op']).size().reset_index(name = 'count')
freqtableofclassandyear = pd.DataFrame(p)    
freqtableofclassandyear


# **Frequency Distribution Table for Survival Status and Axillary Nodes**

# In[288]:


i = haberman.groupby(['survival_status','axillary_nodes']).size().reset_index(name = 'count')
freqtableofclassandaxillary = pd.DataFrame(i)
freqtableofclassandaxillary


# **Bar Graphs**

# In[289]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[290]:


import matplotlib.pyplot as plt
plt.style.available


# In[291]:


import matplotlib.pyplot as plt
freqtableofclasses.plot.bar(width = .3)
plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('seaborn-darkgrid')


# In[292]:


freqtableofclasses[['survival_status']].plot.bar(width = .2)
plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('seaborn-dark-palette')


# In[293]:


freqtableofclasses[['count']].plot.bar(width = .2)
plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('seaborn-poster')


# In[294]:


freqtableofclasses.plot.bar(x = 'survival_status',y = 'count', width = .2)
plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('tableau-colorblind10')


# In[341]:


freqtableofage.plot.bar(width = .7)
plt.rcParams['figure.figsize'] = (20,15)
plt.style.use('seaborn-deep')


# In[296]:


freqtableofage[['age']].plot.bar()
plt.rcParams['figure.figsize'] = (30,20)
plt.style.use('seaborn-darkgrid')


# In[297]:


freqtableofage[['count']].plot.bar()
plt.rcParams['figure.figsize'] = (15,15)
plt.style.use('seaborn-darkgrid')


# In[298]:


freqtableofage.plot.bar(x = 'age', y = 'count')
plt.rcParams['figure.figsize'] = (15,10)
plt.style.use('seaborn-bright')


# In[299]:


freqtableofyear.plot.bar()
plt.rcParams['figure.figsize'] = (12,10)
plt.style.use('seaborn-darkgrid')


# In[300]:


freqtableofyear[['year_of_op']].plot.bar()
plt.rcParams['figure.figsize'] = (25,15)
plt.style.use('seaborn-darkgrid')


# In[301]:


freqtableofyear[['count']].plot.bar()
plt.rcParams['figure.figsize'] = (25,15)
plt.style.use('seaborn-darkgrid')


# In[302]:


freqtableofyear.plot.bar(x = 'year_of_op', y = 'count')
plt.rcParams['figure.figsize'] = (25,15)
plt.style.use('seaborn-darkgrid')


# In[303]:


freqtableofaxils.plot.bar()
plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('seaborn-poster')


# In[304]:


freqtableofaxils[['axillary_nodes']].plot.bar()
plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('seaborn-poster')


# In[305]:


freqtableofaxils[['count']].plot.bar()
plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('seaborn-poster')


# In[306]:


freqtableofaxils.plot.bar(x = 'axillary_nodes', y = 'count')
plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('seaborn-poster')


# **Stacked Bar Graphs**

# In[307]:


import warnings
warnings.filterwarnings('ignore')


# In[383]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')


# In[309]:


import numpy as np
yolo = freqtableofclassandage['age']
arr = yolo.to_numpy()
print(arr)

m = freqtableofclassandage.loc[freqtableofclassandage['survival_status'] == 1 ]
subarr1 = m.to_numpy()
print(m)
print(subarr1)
k = np.array(m.survival_status)
print(k)

n = freqtableofclassandage.loc[freqtableofclassandage['survival_status'] == 2 ]
print(n)
subarr2 = n.to_numpy()
print(subarr2)
j = np.array(n.survival_status)
print(j)


# In[340]:


def aggregate(rows,columns,df):
    column_keys = df[columns].unique()
    row_keys = df[rows].unique()

    agg = { key : [ len(df[(df[rows]==value) & (df[columns]==key)]) for value in row_keys]
               for key in column_keys }

    aggdf = pd.DataFrame(agg,index = row_keys)
    aggdf.index.rename(rows,inplace=True)

    return aggdf

aggregate('age','survival_status',freqtableofclassandage).plot(kind='bar',stacked=True)
plt.rcParams['figure.figsize'] = (20,20)
plt.style.use('seaborn-darkgrid')


# In[311]:


def aggregate(rows,columns,df):
    column_keys = df[columns].unique()
    row_keys = df[rows].unique()

    agg = { key : [ len(df[(df[rows]==value) & (df[columns]==key)]) for value in row_keys]
               for key in column_keys }

    aggdf = pd.DataFrame(agg,index = row_keys)
    aggdf.index.rename(rows,inplace=True)

    return aggdf

aggregate('survival_status','age',freqtableofclassandage).plot(kind='bar',stacked=True)
plt.rcParams['figure.figsize'] = (30,30)
plt.style.use('seaborn-darkgrid')


# In[312]:


aggregate('survival_status','year_of_op',freqtableofclassandyear).plot(kind='bar',stacked=True)
plt.rcParams['figure.figsize'] = (30,20)
plt.style.use('ggplot')


# In[313]:


aggregate('year_of_op','survival_status',freqtableofclassandyear).plot(kind='bar',stacked=True)
plt.rcParams['figure.figsize'] = (30,15)
plt.style.use('ggplot')


# In[314]:


aggregate('axillary_nodes','survival_status',freqtableofclassandaxillary).plot(kind='bar',stacked=True)
plt.rcParams['figure.figsize'] = (20,10)
plt.style.use('ggplot')


# **Histograms and KDE Plots** 

# In[315]:


import seaborn as sn
sn.distplot(haberman['age'], bins = 22, color = 'red')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[316]:


sn.distplot(haberman['age'], bins = 30, kde = False ,color = 'red')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[1]:



sn.distplot(haberman['age'], bins = 22, color = 'red', hist = False,rug = True)
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[318]:


sn.kdeplot(haberman['age'],color = 'red',shade = True)
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[319]:


sn.kdeplot(haberman['age'], shade = True, color = 'green')
sn.kdeplot(haberman['age'],bw = .8, label = 'bw = 0.1', shade = True, color = 'blue')
sn.kdeplot(haberman['age'],bw = 2, label = 'bw = 5', shade = True, color = 'orange')


# In[320]:


sn.kdeplot(haberman['age'],shade = True, cut = 0, color = 'green')
sn.rugplot(haberman['age'])


# In[321]:


sn.distplot(haberman['year_of_op'], bins = 10, color = 'indigo')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[385]:


sn.distplot(haberman['year_of_op'], bins = 12, color = 'indigo',kde = False)
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[323]:


sn.distplot(haberman['year_of_op'],hist = False, bins = 5,rug = True, color = 'indigo')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[324]:


sn.kdeplot(haberman['year_of_op'], shade = True, color = 'indigo')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[325]:


sn.kdeplot(haberman['year_of_op'],shade = True, cut = 0, color = 'brown')
sn.rugplot(haberman['year_of_op'])


# In[326]:


sn.distplot(haberman['axillary_nodes'], bins = 10, color = 'black')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[327]:


sn.distplot(haberman['axillary_nodes'],kde = False, bins = 8, color = 'black')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[328]:


sn.distplot(haberman['axillary_nodes'],hist = False, bins = 5,rug = True, color = 'black')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[329]:


sn.kdeplot(haberman['axillary_nodes'], shade = True, color = 'black')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[330]:


sn.kdeplot(haberman['axillary_nodes'],shade = True, cut = 0, color = 'Blue')
sn.rugplot(haberman['axillary_nodes'])


# In[331]:


sn.distplot(haberman['survival_status'], bins = 10, color = 'darkviolet')
plt.rcParams['figure.figsize'] = (5,5)
plt.style.use('seaborn-darkgrid')


# In[336]:


sn.distplot(haberman['survival_status'],kde = False, bins = 10, color = 'darkviolet')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[337]:


sn.distplot(haberman['survival_status'],hist = False, bins =3,rug = True,color= 'violet')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[334]:


sn.kdeplot(haberman['survival_status'], shade = True, color = 'violet')
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('seaborn-darkgrid')


# In[338]:


sn.kdeplot(haberman['survival_status'],shade = True, cut = 0, color = 'red')
sn.rugplot(haberman['survival_status'])


# **Bivariate Analysis**

# In[354]:


sn.jointplot(x = 'age', y = 'count', data = freqtableofage,color = 'blue' )
plt.style.use('seaborn-darkgrid')


# In[380]:


sn.jointplot(x = 'age', y = 'count', data = freqtableofage, color = 'green',kind = 'reg',ratio = 5) 
plt.style.use('seaborn-darkgrid')


# In[384]:


sn.jointplot(x = 'age', y = 'count',ratio = 10, data = freqtableofage, color = 'black',kind = 'hex') 
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (20,10)


# In[371]:


sn.jointplot(x = 'age', y = 'count', data = freqtableofage, color = 'red',kind = 'kde') 
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (20,10)


# In[386]:


sn.jointplot(x = 'year_of_op', y = 'count', data = freqtableofyear,color = 'black' )
plt.style.use('seaborn-darkgrid')


# In[391]:


sn.jointplot(x = 'year_of_op', y = 'count',kind = 'reg' ,data = freqtableofyear,color = 'orange', ratio = 8 )
plt.style.use('seaborn-darkgrid')


# In[392]:


sn.jointplot(x = 'year_of_op', y = 'count',kind = 'hex' ,data = freqtableofyear,color = 'indigo', ratio = 8 )
plt.style.use('seaborn-darkgrid')


# In[393]:


sn.jointplot(x = 'year_of_op', y = 'count',kind = 'kde' ,data = freqtableofyear,color = 'blue', ratio = 8 )
plt.style.use('seaborn-darkgrid')


# In[398]:


sn.jointplot(x = 'axillary_nodes', y = 'count', data = freqtableofaxils,color = 'black' )
plt.style.use('seaborn-darkgrid')


# In[403]:


sn.jointplot(x = 'axillary_nodes', y = 'count', data = freqtableofaxils,color = 'green',kind = 'reg', ratio = 10)
plt.style.use('seaborn-darkgrid')


# In[405]:


sn.jointplot(x = 'axillary_nodes', y = 'count', data = freqtableofaxils,color = 'orange',kind = 'kde', ratio = 10)
plt.style.use('seaborn-darkgrid')


# In[410]:


sn.jointplot(x= 'age', y = 'survival_status',data = haberman , color = 'violet',ratio = 5)


# In[411]:


sn.jointplot(x= 'age', y = 'survival_status',data = haberman , color = 'green',ratio = 5,kind = 'kde')


# In[413]:


sn.jointplot(x= 'year_of_op', y = 'survival_status',data = haberman , color = 'brown',ratio = 5,kind = 'kde')


# In[414]:


sn.jointplot(x= 'year_of_op', y = 'survival_status',data = haberman , color = 'brown',ratio = 5)


# In[416]:


sn.jointplot(x= 'axillary_nodes', y = 'survival_status',data = haberman , color = 'indigo',ratio = 5,kind = 'kde')


# In[417]:


sn.jointplot(x= 'axillary_nodes', y = 'survival_status',data = haberman , color = 'blue',ratio = 5)


# **Pair Plots**

# In[436]:


curr = sn.color_palette()
sn.palplot(curr)


# In[489]:


op = sn.pairplot(haberman,palette = 'husl')
op.fig.set_size_inches(15,15)


# In[471]:


p = sn.pairplot(haberman, hue = 'survival_status',palette = 'cubehelix')
p.fig.set_size_inches(15,15)


# In[452]:


bv = sn.pairplot(haberman, hue = 'survival_status',kind = 'reg',palette = 'husl')
bv.fig.set_size_inches(15,15)


# In[474]:


rk = sn.pairplot(haberman, hue = 'survival_status', corner =True)
rk.fig.set_size_inches(15,15)


# In[450]:


lp = sn.pairplot(freqtableofclassandage)
lp.fig.set_size_inches(15,15)


# In[466]:


vp = sn.pairplot(freqtableofclassandage, hue = 'survival_status', palette = "ocean")
vp.fig.set_size_inches(15,15)


# In[467]:


kp = sn.pairplot(freqtableofclassandage, hue = 'survival_status',kind = 'reg', palette = 'gist_stern')
kp.fig.set_size_inches(15,15)


# In[476]:


kv = sn.pairplot(freqtableofclassandyear)
kv.fig.set_size_inches(15,15)


# In[478]:


vp = sn.pairplot(freqtableofclassandyear, hue = 'survival_status', palette = "Dark2")
vp.fig.set_size_inches(15,15)


# In[480]:


mv = sn.pairplot(freqtableofclassandyear, hue = 'survival_status', palette = "prism",kind = 'reg')
mv.fig.set_size_inches(15,15)


# In[481]:


rgv = sn.pairplot(freqtableofclassandaxillary)
rgv.fig.set_size_inches(15,15)


# In[484]:


pgv = sn.pairplot(freqtableofclassandaxillary,hue = 'survival_status', palette = "gist_ncar")
pgv.fig.set_size_inches(15,15)


# In[486]:


ac = sn.pairplot(freqtableofclassandaxillary,hue = 'survival_status', palette = "nipy_spectral",kind = 'reg')
ac.fig.set_size_inches(15,15)


# ## **Detect the Distributions of Age Using Normality Tests**

# **Normality Assumption**
# 
# A large fraction of the field of statistics is concerned with data that assumes that it was drawn from a Gaussian distribution.<br>
# 
# If methods are used that assume a Gaussian distribution, and your data was drawn from a different distribution, the findings may be misleading or plain wrong.<br>
# 
# There are a number of techniques that you can check if your data sample is Gaussian or sufficiently Gaussian-like to use the standard techniques, or sufficiently non-Gaussian to instead use non-parametric statistical methods.<br>
# 
# This is a key decision point when it comes to choosing statistical methods for your data sample. We can summarize this decision as follows:<br>

# if Data is Gaussian:<br>
#     print('Use Parametric Statistical Methods')<br>
# else:<br>
#     print('Use Non-Parametric Statistical Methods')

# In[500]:


from numpy.random import seed
from numpy.random import randn
from numpy import mean 
from numpy import std
seed(1)
data = haberman['age']
print('mean = %.3f  std = %.3f'%(mean(data),std(data)))


# **Visual Normality Check using Histogram Plot**

# In[503]:


from matplotlib import pyplot
pyplot.hist(data)
pyplot.show()
plt.rcParams['figure.figsize'] = (5,5)
plt.style.use('seaborn-darkgrid')


# We can see a Gaussian-like shape to the data, that although is not strongly the familiar bell-shape, is a rough approximation.
# 
# 

# **Quantile - Quantile Plot**

# Popular plot for checking the distribution of a data sample is the quantile-quantile plot, Q-Q plot, or QQ plot for short.<br>
# 
# This plot generates its own sample of the idealized distribution that we are comparing with, in this case the Gaussian distribution.<br>
# 
# The idealized samples are divided into groups called quantiles. Each data point in the sample is paired with a similar member from the idealized distribution at the same cumulative distribution.<br> 
# 
# The resulting points are plotted as a scatter plot with the idealized value on the x-axis and the data sample on the y-axis.<br>
# 
# A perfect match for the distribution will be shown by a line of dots on a 45-degree angle from the bottom left of the plot to the top right. Often a line is drawn on the plot to help make this expectation clear. Deviations by the dots from the line shows a deviation from the expected distribution.<br>
# 
# 
# 

# In[510]:


from statsmodels.graphics.gofplots import qqplot
data = haberman['age']
qqplot(data , line = 's')
pyplot.show()
plt.rcParams['figure.figsize'] = (8,8)


# Running the above code creates the QQ plot showing the scatter plot of points in a diagonal line, closely fitting the expected diagonal pattern for a sample from a Gaussian distribution.<br>
# 
# There are a few small deviations, especially at the bottom of the plot, which is to be expected given the small data sample.
# 

# **Statistical Normality Tests**

# There are 3 statistical tests to quantify whether a sample of data looks as though it was drawn from a Gaussian distribution.<br>
# 
# Each test calculates a test-specific statistic. This statistic can aid in the interpretation of the result.<br>
# 
# Instead, the p-value can be used to quickly and accurately interpret the statistic in practical applications.<br>
# 
# The tests assume that that the sample was drawn from a Gaussian distribution. Technically this is called the null hypothesis, or H0. A threshold level is chosen called alpha, typically 5% (or 0.05), that is used to interpret the p-value.<br>
# 
# In the SciPy implementation of these tests, you can interpret the p value as follows.<br<
# 
# **p <= alpha**: reject H0, not normal.<br>
# **p > alpha**: fail to reject H0, normal.<br>
# 
# This means that, in general, we are seeking results with a larger p-value to confirm that our sample was likely drawn from a Gaussian distribution.<br>
# 
# A result above 5% does not mean that the null hypothesis is true. It means that it is very likely true given available evidence. The p-value is not the probability of the data fitting a Gaussian distribution; it can be thought of as a value that helps us interpret the statistical test.
# 

# **Shapiro-Wilk Test**

# In[514]:


from scipy.stats import shapiro
data = haberman['age']

# Do the Shapiro Normality Test

stat, p = shapiro(data)
print('stats = %.3f , p = %.3f'%(stat,p))

# Interpret at alpha = 0.05
if p>0.05:
    print('Data Looks Gauusian(fail to reject H0)')
else:
    print('Data Does not look Gaussian(reject H0)')


# **D’Agostino’s K^2 Test**
# 
# The **D’Agostino’s K^2 Test**calculates summary statistics from the data, namely kurtosis and skewness, to determine if the data distribution departs from the normal distribution, named for Ralph D’Agostino.<br>
# 
# Skew is a quantification of how much a distribution is pushed left or right, a measure of asymmetry in the distribution.<br>
# 
# Kurtosis quantifies how much of the distribution is in the tail. It is a simple and commonly used statistical test for normality.

# In[515]:


from scipy.stats import normaltest
data = haberman['age']

# Do the general normality Test on Data

stat, p = normaltest(data)
print('stats = %.3f, p = %.3f'%(stat,p))

# Interpret at alpha = 0.05
if p>0.05:
    print('Data Looks Gauusian(fail to reject H0)')
else:
    print('Data Does not look Gaussian(reject H0)')


# **Anderson-Darling Test**
# 
# Anderson-Darling Test is a statistical test that can be used to evaluate whether a data sample comes from one of among many known data samples, named for Theodore Anderson and Donald Darling.<br>
# 
# It can be used to check whether a data sample is normal. The test is a **modified version of** a more sophisticated nonparametric goodness-of-fit statistical test called the **Kolmogorov-Smirnov test**.<br>
# 
# It takes as parameters the data sample and the name of the distribution to test it against. By default, the test will check against the Gaussian distribution (dist=’norm’).

# In[516]:


from scipy.stats import anderson
data = haberman['age']

# Do the Normality Test

result = anderson(data)
print('Test statistic = %.3f, p = %.3f'% re)

