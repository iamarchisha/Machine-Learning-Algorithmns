# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data = np.genfromtxt(path, delimiter=",", skip_header=1)
#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
census = np.concatenate((data,new_record),axis = 0)


# --------------
#Code starts here
age = census[:,0]
print(age)
max_age = age.max()
min_age = age.min()
age_mean = age.mean()
age_std = np.std(age)


# --------------
#Code starts here
race = census[:,2].astype(np.int32)
race0 = (race ==0)
race_0 = census[race0]
race1 = (race==1)
race_1 = census[race1]
race2 = (race==2)
race_2 = census[race2]
race3 = (race ==3)
race_3 = census[race3]
race4 = (race==4)
race_4 = census[race4]
print(race_3)


len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)
print(len_0,len_1,len_2,len_3,len_4)
len_array = np.array([len_0,len_1,len_2,len_3,len_4])

minority_race = len_array.argmin()
print(minority_race)
max_citizens = len_array.max()


# --------------
#Code starts here
senior_citizens = census[census[:,0]>60].astype(np.int64)

working_hours_sum = senior_citizens[:,6].sum()
print('all the working hours of senior citizens',working_hours_sum)

senior_citizens_len = len(senior_citizens)

avg_working_hours = working_hours_sum/senior_citizens_len
print('average working hours: ',avg_working_hours)


# --------------
#Code starts here

#partitioning data into high and low with respect to education-num
high = census[census[:,1]>10].astype(np.int64)
low = census[census[:,1]<=10].astype(np.int64)

#finding the mean of income column for high and low
avg_pay_high = high[:,7].mean()
avg_pay_low = low[:,7].mean()

#displaying the average income for low and high
print('average income for low: ',avg_pay_low)
print('average income for high: ',avg_pay_high)


