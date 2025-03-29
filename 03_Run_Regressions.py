# Written by James Cabral and Chris Haun
# This program runs regressions

#NOTE: Some of the implementation follows the steps from the replication code of Bisbee et al. (2021)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import statsmodels.formula.api as smf

os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\Winter\ECO2460\Empirical Project\Data")

reg_data = pd.read_csv("Final_Data_Reg.csv")

####
# Merge in the Economic Indicators

#EPU Index
#Retrieved from: 
# https://www.policyuncertainty.com/canada_monthly.html
#deleted bottom column manually
EPU_data = pd.read_excel("Canada_Policy_Uncertainty_Data.xlsx", sheet_name= 'Canadian News-Based Index')
EPU_data = EPU_data.rename(columns={
    "Canada News-Based Policy Uncertainty Index":"EPU"})

reg_data['date'] = pd.to_datetime(reg_data['date'])

# Extract month and year
reg_data['Year'] = reg_data['date'].dt.year
reg_data['Month'] = reg_data['date'].dt.month

reg_data = pd.merge(reg_data, EPU_data, how='left', on=['Year', 'Month'])

topic_columns = reg_data.filter(regex='^topic_\d+$').columns
topic_columns = topic_columns[1:]

mention_columns = reg_data.filter(regex='^mention_').columns
mention_columns = [col for col in mention_columns if col != 'mention_any']

formula = "SENT_Hostile ~ gender + EPU + mention_any + EPU * mention_any + " + \
          " + ".join(topic_columns) + \
          " + C(party) + C(Year)"
         
model = smf.ols(formula=formula, data=reg_data).fit()
print(model.summary())


formula = "SENT_Hostile ~ gender + EPU " + \
          " + ".join(mention_columns) + " + " + \
          " + ".join(topic_columns) + \
          " + C(party) + C(Year)"
         
model = smf.ols(formula=formula, data=reg_data).fit()
print(model.summary())
