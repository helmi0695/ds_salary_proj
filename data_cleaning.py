# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:30:25 2021

@author: HB6
"""

import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

# salary parsing :eleminate the values -1/  remove the () / keep only the min-max sal/ calculate avg sal
# create a column where per hour takes 0 and yearly takes 1
df['hourly'] = df["Salary Estimate"].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df["Salary Estimate"].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

df = df[df["Salary Estimate"] != '-1'] 
sal_clean =df["Salary Estimate"].apply(lambda x : x.split('(')[0])
sal_clean = sal_clean.apply(lambda x: x.replace('$','').replace('K',''))

sal_clean = sal_clean.apply(lambda x : x.replace('Per Hour','').replace('Employer Provided Salary:','').strip())

df['min_sal'] = sal_clean.apply(lambda x : int(x.split('-')[0]))
df['max_sal'] = sal_clean.apply(lambda x : int(x.split('-')[1]))

df['avg_sal'] = (df['min_sal'] + df['max_sal'])/2

# clean the company name
df["company_txt"] = df.apply(lambda x : x['Company Name'] if x['Rating'] == -1 else x['Company Name'][:-3], axis=1)

# location + HQ keep only the state
df['job_state'] = df['Location'].apply(lambda x : x.split(',')[1].strip())
df['job_state'].value_counts()

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)

# age of company
df['age'] = df.Founded.apply(lambda x: 2020-x if x != -1 else x)


# parse the job description (data sc tools: python, Rstudio...)




# create a column to see if the location is the same as the HQ
# clean the size
#column to see the age of the company
# clean the revenu




