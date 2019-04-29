# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
p_a = (len(df.loc[df['fico'] > 700]))/len(df)
p_b = (len(df.loc[df.purpose == 'debt_consolidation']))/len(df)
df1 = df.loc[df.purpose == 'debt_consolidation']
p_a_b = ( p_a + p_b )/p_a
p_b_a = ( p_a + p_b )/p_b
result = (p_b_a == p_a)
print(result)
# code ends here


# --------------
# code starts here
from sklearn.naive_bayes import GaussianNB

prob_lp = (len(df.loc[df['paid.back.loan'] == 'Yes']))/len(df)
prob_cs = (len(df.loc[df['credit.policy'] == 'Yes']))/len(df)
new_df = df.loc[df['paid.back.loan'] == 'Yes']

prob_pd_cs = len(new_df[new_df['credit.policy'] == 'Yes'])/len(new_df)

print(prob_pd_cs)
bayes=(prob_pd_cs*prob_lp)/prob_cs
print(bayes)

# code ends here


# --------------
# code starts here
plt.bar(len(df.purpose),height=100,width=100)
df1 = df[df['paid.back.loan'] == 'No']
#df1 = df.loc(df['paid.back.loan']=='No').shape[0]
plt.bar(df1['purpose'],height=100,width=100)
# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
inst_mean = df['installment'].mean()
plt.hist(df['installment'], normed=True, bins=30)
plt.hist(df['log.annual.inc'], normed=True, bins=30)
# code ends here


