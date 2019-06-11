# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here

#loading the dataset
data = pd.read_csv(path)

#plotting a histogram to see distribution of app ratings
plt.hist(data['Rating'].dropna(),alpha=0.9, color='blue')

#cleaning the data to have ratings<=5 only
data = data[data['Rating']<=5]

#plotting histogram to see distribution of app ratings
plt.hist(data['Rating'].dropna())

#Code ends here


# --------------
# code starts here

#counting null values in each column
total_null = data.isnull().sum()
percent_null = total_null/data.isnull().count()

#total missing data
missing_data = pd.concat([total_null,percent_null],keys=['Total','Percent'],axis=1)
print(missing_data)

#dropping null values
data.dropna(inplace=True)

#recalculating null values
total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1/data.isnull().count()

#checking for missing data
missing_data_1 = pd.concat([total_null_1,percent_null_1],keys=['Total','Percent'],axis=1)
print(missing_data_1)

# code ends here


# --------------

#Code starts here
import seaborn as sns

#plotting catplot between Category and Rating
sns.set(style="ticks")
plot = sns.catplot(x="Category",y="Rating",data=data, kind="box",height = 10)
plot.set_xticklabels(rotation=90)
plot.set_titles('Rating vs Category [BoxPlot]')
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

#Code starts here

#checking distribution of column Installs
print(data['Installs'])

#removing '+' and ',' from Installs
data['Installs']=data['Installs'].apply(lambda x: x.replace(',', ''))
data['Installs']=data['Installs'].apply(lambda x: x.replace('+', ''))

#convert dtype of Installs to int
data['Installs'] = data['Installs'].astype(np.int64)
print(data['Installs'])

#transforming values of Installs column using LabelEncoder
le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])

#plotting a regplot between Installs and Rating
sns.set(color_codes=True)
regplt = sns.regplot(x="Installs",y="Rating",data=data)
regplt.set_title('Rating vs Installs [RegPlot]')

#Code ends here



# --------------
#Code starts here

#checking distribution of Price column
print(data['Price'])

#removing '$' from Price
data['Price']=data['Price'].apply(lambda x: x.replace('$', ''))

#convert dtype of Price to int
data['Price'] = data['Price'].astype(np.float64)
print(data['Price'])

#plotting a regplot between Price and Rating
sns.set(color_codes=True)
regplt = sns.regplot(x="Price",y="Rating",data=data)
regplt.set_title('Rating vs Price [RegPlot]')

#Code ends here


# --------------

#Code starts here

#checking for unique values in Genres column
data['Genres'].unique()

#storing only the first genre in Genre column
data['Genres'] = data['Genres'].apply(lambda x: x.split(';',1)[0])

#grouping Genres and Rating by Genres
gr_mean = data.groupby(['Genres'],as_index = False)['Rating'].mean()
print(gr_mean)

#observing statistics of gr_mean
print(gr_mean.describe())

#sorting gr_mean 
gr_mean = gr_mean.sort_values(by =['Rating'], ascending=True)
print(gr_mean)

#printing 1st and last value of gr_mean
print(gr_mean.iloc[1],gr_mean.iloc[-1])


#Code ends here


# --------------

#Code starts here

#visualising values of Last Updated column
print(data['Last Updated'])

#converting data type of Last Updated to datetime format
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

#finding maximum value in Last Updated column
max_date = data['Last Updated'].max()

#creating a new column: Last Updated Days
diff_dates = max_date - data['Last Updated']
data['Last Updated Days'] = diff_dates.dt.days

#plotting reglot between Last Updated Days and Rating
sns.set(color_codes=True)
regplt = sns.regplot(x="Last Updated Days",y="Rating",data=data)
regplt.set_title('Rating vs Last Updated Days [RegPlot]')


#Code ends here


