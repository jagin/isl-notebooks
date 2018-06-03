
# coding: utf-8

# ## 3.6 Lab - Linear Regression

# ### 3.6.1 Libraries

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

from collections import OrderedDict

plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size = 14)
plt.rc('figure', titlesize = 18)
plt.rc('axes', labelsize = 15)
plt.rc('axes', titlesize = 18)

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 3.6.2 Simple Linear Regression

# In[2]:


df = pd.read_csv('../../../data/Boston.csv', header = 0)
df.head()


# In[3]:


# list(df)
df.columns


# In[4]:


df.shape


# **Using statsmodels**

# In[5]:


# predictor & dependent var
x_train = df['lstat']
y_true = df['medv']

# ols model with intercept added to predictor
lm = sm.OLS(y_true, sm.add_constant(x_train))

# fitted model and summary
lm_fit = lm.fit()
lm_fit.summary()

# robust SE
# lm = sm.RLM(y_true, x_train, M=sm.robust.norms.LeastSquares())
# lm_fit = lm.fit(cov='H2')
# lm_fit.summary()


# or

# In[6]:


lm = smf.ols('medv~lstat', data = df)
lm_fit = lm.fit()
lm_fit.summary()


# In[7]:


dir(lm_fit)


# In[8]:


lm_fit.params


# In[9]:


lm_fit.conf_int()


# In[10]:


x_test = pd.DataFrame({'lstat': [5, 10, 15]})
y_pred = lm_fit.get_prediction(x_test)
y_pred.summary_frame()


# In[11]:


x_test = pd.DataFrame({'lstat': [df.lstat.min(), df.lstat.max()]})
y_pred = lm_fit.predict(x_test)

df.plot(x = 'lstat', y = 'medv', kind = 'scatter')
plt.plot(x_test, y_pred, c = 'red', linewidth = 2)
plt.xlabel("lstat")
plt.ylabel("medv")
plt.show()


# In[12]:


x_test = df['lstat']
y_pred = lm_fit.predict(x_test)

f, axes = plt.subplots(1, 2, sharex = False, sharey = False) 
f.set_figheight(10)
f.set_figwidth(15)

sns.regplot('lstat', 'medv', data = df, ax=axes[0], scatter_kws = {'alpha': '0.5'})
sns.residplot(y_pred, 'medv', data=df, ax=axes[1], scatter_kws={'alpha': '0.5'}, lowess = True) # residual plot
axes[1].set_xlabel('Fitted values')
axes[1].set_ylabel('Residuals')
plt.show()


# For creating diagnostic plots see [Emulating R regression plots in Python](https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/)

# In[13]:


# fitted values (need a constant term for intercept)
lm_fitted_y = lm_fit.fittedvalues

# model residuals
lm_residuals = lm_fit.resid

# normalized residuals
lm_norm_residuals = lm_fit.get_influence().resid_studentized_internal

# absolute squared normalized residuals
lm_norm_residuals_abs_sqrt = np.sqrt(np.abs(lm_norm_residuals))

# absolute residuals
lm_abs_resid = np.abs(lm_residuals)

# leverage, from statsmodels internals
lm_leverage = lm_fit.get_influence().hat_matrix_diag

# cook's distance, from statsmodels internals
lm_cooks = lm_fit.get_influence().cooks_distance[0]


# **Residual plot**

# In[14]:


plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sns.residplot(lm_fitted_y, 'medv', data = df,
                                  lowess = True,
                                  scatter_kws = {'alpha': 0.5},
                                  line_kws = {'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')


# annotations
abs_resid = lm_abs_resid.sort_values(ascending = False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_lm_1.axes[0].annotate(i, xy = (lm_fitted_y[i], lm_residuals[i]));


# **QQ plot**

# In[15]:


QQ = ProbPlot(lm_norm_residuals)
plot_lm_2 = QQ.qqplot(line = '45', alpha = 0.5, color = '#4C72B0', lw = 1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(lm_norm_residuals)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], lm_norm_residuals[i]));


# **Scale-Location Plot**

# In[16]:


plot_lm_3 = plt.figure(3)
plot_lm_3.set_figheight(8)
plot_lm_3.set_figwidth(12)

plt.scatter(lm_fitted_y, lm_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(lm_fitted_y, lm_norm_residuals_abs_sqrt, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

# annotations
abs_sq_norm_resid = np.flip(np.argsort(lm_norm_residuals_abs_sqrt), 0)
abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]

for i in abs_norm_resid_top_3:
    plot_lm_3.axes[0].annotate(i, xy=(lm_fitted_y[i], lm_norm_residuals_abs_sqrt[i]));


# **Leverage plot**

# In[17]:


plot_lm_4 = plt.figure(4)
plot_lm_4.set_figheight(8)
plot_lm_4.set_figwidth(12)

plt.scatter(lm_leverage, lm_norm_residuals, alpha=0.5)
sns.regplot(lm_leverage, lm_norm_residuals, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_4.axes[0].set_xlim(0, 0.20)
plot_lm_4.axes[0].set_ylim(-3, 5)
plot_lm_4.axes[0].set_title('Residuals vs Leverage')
plot_lm_4.axes[0].set_xlabel('Leverage')
plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

# annotations
leverage_top_3 = np.flip(np.argsort(lm_cooks), 0)[:3]

for i in leverage_top_3:
    plot_lm_4.axes[0].annotate(i, xy=(lm_leverage[i], lm_norm_residuals[i]))
    
# shenanigans for cook's distance contours
def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

p = len(lm_fit.params) # number of model parameters

graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50), 
      'Cook\'s distance') # 0.5 line

graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50)) # 1 line

plt.legend(loc='upper right');


# ### 3.6.3 Multiple Linear Regression

# In[18]:


lm_fit = smf.ols('medv~lstat+age', data = df).fit()
lm_fit.summary()


# If we want to use all the variable. We can use the following trick to manually construct the list. In Python, most of time, you have to manully construct the variable list.

# In[19]:


def ols_formula(df, dependent_var, *excluded_cols):
    '''
    Generates the R style formula for statsmodels (patsy) given
    the dataframe, dependent variable and optional excluded columns
    as strings
    '''
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    return dependent_var + ' ~ ' + ' + '.join(df_columns)


# In[20]:


lm_fit = smf.ols(formula = ols_formula(df, 'medv'), data = df).fit() # formula = 'medv ~ .'
lm_fit.summary()


# Variance inﬂation factors

# In[21]:


# don't forget to add constant if the ols model includes intercept
df_exog = sm.add_constant(df.drop('medv', axis = 1))

# too fancy for printing results?
for i, col in enumerate(df.columns):
    if col == 'const':
        pass
    elif len(col) > 6:
        print(col, ':', "{0:.2f}".format(vif(df_exog.as_matrix(), i)))
    else:
        print(col, '\t:', "{0:.2f}".format(vif(df_exog.as_matrix(), i)))


#  Run a regression excluding *age* predictor (formula = 'medv ~ . - age')

# In[22]:


lm = smf.ols(formula = ols_formula(df, 'medv', 'age'), data = df)
lm_fit = lm.fit()
lm_fit.summary()


# ### 3.6.4 Interaction Terms

# In[23]:


lm = smf.ols(formula = 'medv ~ lstat * age', data = df)
lm_fit = lm.fit()
lm_fit.summary()


# ### 3.6.5 Non-linear Transformations of Predictors

# In[24]:


lm = smf.ols(formula = 'medv ~ lstat + I(lstat ** 2.0)', data = df)
lm_fit = lm.fit()
lm_fit.summary()


# In[25]:


# anova of the two models
lm_fit1 = smf.ols(formula='medv ~ lstat', data = df).fit()
lm_fit2 = smf.ols(formula='medv ~ lstat + I(lstat**2.0)', data = df).fit()

sm.stats.anova_lm(lm_fit1, lm_fit2)


# In[26]:


f, axes = plt.subplots(4, 2, sharex = False, sharey = False)
f.set_figheight(20)

sns.regplot('lstat', 'medv', data=df, ax=axes[0, 0], order=1, line_kws={'color': 'gray'}, scatter_kws={'alpha': '0.5'})
sns.regplot('lstat', 'medv', data=df, ax=axes[0, 1], order=2, line_kws={'color': 'gray'}, scatter_kws={'alpha': '0.5'})
sns.regplot('lstat', 'medv', data=df, ax=axes[1, 0], order=3, line_kws={'color': 'gray'}, scatter_kws={'alpha': '0.5'})
sns.regplot('lstat', 'medv', data=df, ax=axes[1, 1], order=5, line_kws={'color': 'gray'}, scatter_kws={'alpha': '0.5'})
sns.residplot('lstat', 'medv', data=df, ax=axes[2, 0], order=1, line_kws={'color': 'gray'}, scatter_kws={'alpha': '0.5'}, lowess = True)
sns.residplot('lstat', 'medv', data=df, ax=axes[2, 1], order=2, line_kws={'color': 'gray'}, scatter_kws={'alpha': '0.5'}, lowess = True)
sns.residplot('lstat', 'medv', data=df, ax=axes[3, 0], order=3, line_kws={'color': 'gray'}, scatter_kws={'alpha': '0.5'}, lowess = True)
sns.residplot('lstat', 'medv', data=df, ax=axes[3, 1], order=5, line_kws={'color': 'gray'}, scatter_kws={'alpha': '0.5'}, lowess = True);


# In[27]:


lm = smf.ols(formula = 'medv ~ lstat + I(lstat**2) + I(lstat**3) + I(lstat**4) + I(lstat**5)', data = df)
lm_fit = lm.fit()
lm_fit.summary()


# In[28]:


lm = smf.ols(formula = 'medv ~ np.log(rm)', data = df)
lm_fit = lm.fit()
lm_fit.summary()


# In[29]:


sns.regplot('rm', 'medv', data = df, logx = True, line_kws = {'color': 'gray'}, scatter_kws = {'alpha': '0.5'});


# ### 3.6.6 Qualitative Predictors

# In[30]:


df = pd.read_csv('../../../data/Carseats.csv')
df.head()


# In[31]:


formula = ols_formula(df, 'Sales', 'ShelveLoc') + ' + Income:Advertising + Price:Age + C(ShelveLoc)'
lm = smf.ols(formula, data = df)
lm_fit = lm.fit()
lm_fit.summary()


# ### 3.6.7 Writing Functions

# We already did.
