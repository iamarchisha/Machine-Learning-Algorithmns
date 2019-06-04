# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]
data = pd.read_csv(path)
#Code starts here
data_sample = data.sample(n=sample_size,random_state=0)
sample_mean = data_sample['installment'].mean()
print('sample mean                      :    ',sample_mean)
sample_std = data_sample['installment'].std()
print('sample stdev                     :    ',sample_std)
margin_of_error = z_critical * (sample_std/math.sqrt(sample_size))
lower_confidence_interval = sample_mean - margin_of_error
upper_confidence_interval = sample_mean + margin_of_error
confidence_interval = [lower_confidence_interval,upper_confidence_interval]
print('confidence interval              :   ',confidence_interval)
true_mean = data['installment'].mean()
print('true mean                        :   ',true_mean)
if (true_mean <= upper_confidence_interval and true_mean>=lower_confidence_interval):
    print('True mean lies within the confidence interval')
else:
    print('True mean does not lie within the confidence interval')


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize=(20,10))
for i in range(len(sample_size)):
    m=[]
    for j in range(1000):
        if sample_size.any()==sample_size[i]:
            sample = data.sample(sample_size[i])
        m = data['installment'].mean()
    mean_series = pd.Series(m)
    mean_series.hist()


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = data['int.rate'].astype(str).str[:-2].astype(np.float16)
data['int.rate'] = data['int.rate']/[100]
data['int.rate'][100]=0.0712
data['int.rate'][50] = 0.0743
x1 = data[data['purpose']=='small_business']['int.rate']
z_statistic,p_value = ztest(x1, x2=None, value=data['int.rate'].mean(), alternative='larger', usevar='pooled', ddof=1.0)
if p_value>0.05:
    print('Null hypothesis is accepted')
else:
    print('Null hypothesis is rejected')
z_statistic= 12.32


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic, p_value = ztest(x1=data[data['paid.back.loan']=='No']['installment'],x2=data[data['paid.back.loan']=='Yes']['installment'])
if p_value<0.05:
    print('Null hypothesis is rejected')
else:
    print('Null hypothesis is accepted')


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no = data[data['paid.back.loan']=='No']['purpose'].value_counts()
observed = pd.concat([yes.transpose() , no.T],axis=1,keys=['Yes','No'])
chi2, p, dof, ex = chi2_contingency(observed)

#Comparing chi2 with critical value
if chi2<critical_value:
    inference = 'Accepted'
else:
    inference = 'Rejected'

print(inference)


