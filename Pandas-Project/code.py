# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.DataFrame(pd.read_csv(path))
categorical_var = bank.select_dtypes(include = 'object')
print('categorical variables are: ',categorical_var)
numerical_var = bank.select_dtypes(include = 'number')
print('numerical variables are: ',numerical_var)

# code ends here


# --------------
# code starts here
banks = bank.drop('Loan_ID',1)
print('null values: ', banks.isnull().sum())
bank_mode = banks.mode()
banks.fillna(banks.mode().iloc[0],inplace = True)  
print('null values: ', banks.isnull().sum())
#code ends here


# --------------
# Code starts here





avg_loan_amount = banks.pivot_table(index = ['Gender', 'Married', 'Self_Employed'],values='LoanAmount',aggfunc = 'mean')



# code ends here



# --------------
# code starts here
loan_approved_df = banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')]
loan_approved_se = len(loan_approved_df)
print('ount of results where Self_Employed == Yes and Loan_Status == Y: ',loan_approved_se)

loan_approved_ndf = banks[(banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')]
loan_approved_nse = len(loan_approved_ndf)

percentage_se = (loan_approved_se/614)*100
percentage_nse = (loan_approved_nse/614)*100
print('percentage of loan approval for self employed people: ',percentage_se)
print('percentage of loan approval for not self employed people: ',percentage_nse)
# code ends here


# --------------
# code starts here
loan_term = banks['Loan_Amount_Term'].apply(lambda x: x/12)
big_loan_term = len(banks[loan_term>=25])

print(' number of applicants having loan amount term greater than or equal to 25 years', big_loan_term)


# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby('Loan_Status')
#loan_groupby_df = pd.DataFrame(loan_groupby)
loan_groupby = loan_groupby['ApplicantIncome','Credit_History']
mean_values = loan_groupby.mean()
print(mean_values)
# code ends here


