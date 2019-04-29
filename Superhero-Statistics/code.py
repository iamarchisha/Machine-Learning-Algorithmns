# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
#Code starts here 
data['Gender'].replace('-','Agender',inplace = True)
gender_count = data.Gender.value_counts()
gender_count.plot(kind = 'bar')


# --------------
#Code starts here
alignment = data.Alignment.value_counts()
alignment.plot(kind = 'pie', labels = 'Character Alignment')


# --------------
#Code starts here
sc_df = data.loc[: ,['Strength','Combat']]
sc_covariance = sc_df.Strength.cov(sc_df.Combat)
sc_strength = sc_df.Strength.std()
sc_combat = sc_df.Combat.std()
sc_pearson = sc_covariance/(sc_strength*sc_combat)

ic_df = data.loc[: ,['Intelligence','Combat']]
ic_covariance = ic_df.Intelligence.cov(ic_df.Combat)
ic_intelligence = ic_df.Intelligence.std()
ic_combat = ic_df.Combat.std()
ic_pearson = ic_covariance/(ic_intelligence*ic_combat)


# --------------
#Code starts here
total_high = data.Total.quantile(0.99)
super_best = data.loc[(data.Total > total_high),['Total','Name']]
print(super_best)
a = (super_best.Name)
b = (data.Name)
super_best_names = [i for i in data if ('a' == 'b')]
print(super_best_names)



# --------------
#Code starts here
ax_1 = data.boxplot(column = ['Intelligence'])
ax_2 = data.boxplot(column = ['Speed'])
ax_3 = data.boxplot(column = ['Power'])

ax_1.set_title('Intelligence')
ax_2.set_title('Speed')
ax_3.set_title('Power')


