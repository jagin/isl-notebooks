
# coding: utf-8

# ## 2.3 Lab: Introduction to Python

# ### 2.3.1 Basic Commands

# In[1]:


import numpy as np  # for calculation purpose, let use np.array 
import random # for the random 


# In[2]:


x = np.array([1, 3, 2, 5])


# In[3]:


print(x)


# In[4]:


x = np.array([1, 6, 2])


# In[5]:


print(x)


# In[6]:


y = np.array([1, 4, 3])


# In[7]:


len(x)


# In[8]:


len(y)


# In[9]:


print(x + y)


# In[10]:


get_ipython().run_line_magic('whos', '')


# In[11]:


del x # reset_selective x


# In[12]:


get_ipython().run_line_magic('whos', '')


# In[13]:


get_ipython().run_line_magic('pinfo', 'reset')


# In[14]:


x = np.array([1, 2, 3, 4])
x = np.reshape(x, (2, 2), order='F')


# In[15]:


print(x)


# In[16]:


x = np.array([1, 2, 3, 4])
x = np.reshape(x, (2, 2))


# In[17]:


print(x)


# In[18]:


x = np.matrix([[1, 2], [3, 4]])


# In[19]:


print(x)


# In[20]:


print(np.sqrt(x))


# In[21]:


print(x**2)


# In[22]:


print(np.square(x))


# In[23]:


mu, sigma = 0, 1 # mean and standard deviation


# In[24]:


x = np.random.normal(mu, sigma, 50)


# In[25]:


y = x + np.random.normal(50, 0.1, 50)


# In[26]:


print(np.corrcoef(x, y))


# In[27]:


np.random.seed(1303)


# In[28]:


print(np.random.normal(mu, sigma, 50)) # after set up the seed, this should genernate the same result


# In[29]:


np.random.seed(1303)
y = np.random.normal(mu, sigma, 100)


# In[30]:


print(np.mean(y))


# In[31]:


print(np.var(y))


# In[32]:


print(np.sqrt(np.var(y)))


# In[33]:


print(np.std(y))


# ### 2.3.2 Graphics

# In[34]:


# In python, matplotlib is the most used library for plot 
# matplotlib.pyplot is a collection of command style functions that make matplotlib work like MATLAB.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)


# In[36]:


plt.plot(x, y, 'bo') # please use plt.plot? to look at more options 
plt.ylabel("this is the y-axis")
plt.xlabel("this is the x-axis")
plt.title("Plot of X vs Y")
plt.savefig('../../../output/Figure.pdf') # use plt.savefig function to save images
plt.show()


# In[37]:


x = np.arange(1, 11) # note the arange excludes right end of range specification


# In[38]:


print(x)


# In[39]:


# in order to use Pi, math module needs to be loaded first
import math
x = np.linspace(-math.pi, math.pi, num = 50)


# In[40]:


print(x)


# In[41]:


y = x
X, Y = np.meshgrid(x,y)


# In[42]:


f = np.cos(Y)/(1 + np.square(X))
CS = plt.contour(X, Y, f)
plt.show()


# In[43]:


fa = (f - f.T)/2 #f.T for transpose or tranpose(f)
plt.imshow(fa, extent=(x[0], x[-1], y[0], y[-1])) 
plt.show()


# In[44]:


from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_wireframe(X, Y, fa)
plt.show()


# ### 2.3.3 Indexing Data 

# In[45]:


A = np.arange(1,17,1).reshape(4, 4).transpose()


# In[46]:


print(A)


# In[47]:


# R starts the index from 1, but Python starts the index from 0.
# To select the same number (10) as the book did, we need to reduce the index by 1
print(A[1, 2])


# In[48]:


print(A[[[0],[2]], [1,3]])


# In[49]:


print(A[0:3, 1:4])


# In[50]:


print(A[0:2, :])


# In[51]:


print(A[:, 0:2])


# In[52]:


print(A[0,:])


# In[53]:


# minus sign has a different meaning in Python.
# This means index from the end.
# -1 means the last element
print(A[-1, -1])


# In[54]:


A.shape


# ### 2.3.4 Loading Data

# In[55]:


# In Python, Pandas is a common used module to read from file into a data frame.
import pandas as pd 
Auto = pd.read_csv('../../../data/Auto.csv', header=0, na_values='?')


# In[56]:


Auto.head()


# In[57]:


Auto.shape


# In[58]:


Auto.iloc[32]


# In[59]:


Auto.iloc[:4, :2]


# In[60]:


Auto.columns


# In[61]:


list(Auto)


# In[62]:


# Use .isnull and .sum to find out how many NaNs in each variables
Auto.isnull().sum()


# In[63]:


# There are 397 rows in the data and only 5 with missing values.
# We can just drop the ones with missing values.
Auto = Auto.dropna()


# In[64]:


Auto.shape


# ### 2.3.5 Additional Graphical and Numerical Summaries

# In[65]:


plt.plot(Auto.cylinders, Auto.mpg, 'ro')
plt.show()


# In[66]:


Auto.hist(column = ['cylinders', 'mpg'])
plt.show()


# In[67]:


import seaborn as sns


# In[68]:


sns.pairplot(Auto)
plt.show()


# In[69]:


sns.pairplot(Auto, vars = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration'])
plt.show()


# In[70]:


Auto.describe()


# In[71]:


Auto.describe(include = 'all')


# In[72]:


# Change the cylinders into categorical variable
Auto['cylinders'] = Auto['cylinders'].astype('category')


# In[73]:


Auto.describe(include= 'all')

