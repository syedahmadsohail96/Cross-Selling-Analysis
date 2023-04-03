#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 19:19:29 2023

@author: syedahmadsohail
"""

import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

df1 = pd.read_csv('/Users/syedahmadsohail/Desktop/Study/Project Data Glacier/Test.csv')
print(df1)


df2 = pd.read_csv('/Users/syedahmadsohail/Desktop/Study/Project Data Glacier/Train.csv')
print(df2)

#dictionary to map the old column names to the new ones
new_column_names = {
    'fecha_dato': 'Date',
    'ncodpers': 'Customer_Code',
    'ind_empleado': 'Employee_Index',
    'pais_residencia': 'Country_Residence',
    'sexo': 'Gender',
    'age': 'Age',
    'fecha_alta': 'Account_Opening_Date',
    'ind_nuevo': 'New_Customer_Index',
    'antiguedad': 'Seniority_Months',
    'indrel': 'Primary_Customer_Index',
    'ult_fec_cli_1t': 'Last_Date_As_Primary_Customer',
    'indrel_1mes': 'Customer_Type_At_Beginning_Of_Month',
    'tiprel_1mes': 'Customer_Relation_Type_At_Beginning_Of_Month',
    'indresi': 'Residence_Index',
    'indext': 'Foreigner_Index',
    'conyuemp': 'Spouse_Index',
    'canal_entrada': 'Channel_Used_To_Join',
    'indfall': 'Deceased_Index',
    'tipodom': 'Address_Type',
    'cod_prov': 'Province_Code',
    'nomprov': 'Province_Name',
    'ind_actividad_cliente': 'Activity_Index',
    'renta': 'Gross_Income',
    'segmento': 'Segmentation',
    'ind_ahor_fin_ult1': 'Saving_Account',
    'ind_aval_fin_ult1': 'Guarantees',
    'ind_cco_fin_ult1': 'Current_Accounts',
    'ind_cder_fin_ult1': 'Derivada_Account',
    'ind_cno_fin_ult1': 'Payroll_Account',
    'ind_ctju_fin_ult1': 'Junior_Account',
    'ind_ctma_fin_ult1': 'Mas_Particular_Account',
    'ind_ctop_fin_ult1': 'Particular_Account',
    'ind_ctpp_fin_ult1': 'Particular_Plus_Account',
    'ind_deco_fin_ult1': 'Short_Term_Deposits',
    'ind_deme_fin_ult1': 'Medium_Term_Deposits',
    'ind_dela_fin_ult1': 'Long_Term_Deposits',
    'ind_ecue_fin_ult1': 'E_Account',
    'ind_fond_fin_ult1': 'Funds',
    'ind_hip_fin_ult1': 'Mortgage',
    'ind_plan_fin_ult1': 'Pensions',
    'ind_pres_fin_ult1': 'Loans',
    'ind_reca_fin_ult1': 'Taxes',
    'ind_tjcr_fin_ult1': 'Credit_Card',
    'ind_valo_fin_ult1': 'Securities',
    'ind_viv_fin_ult1': 'Home_Account',
    'ind_nomina_ult1': 'Payroll',
    'ind_nom_pens_ult1': 'Pensions_2',
    'ind_recibo_ult1': 'Direct_Debit'
}

# Rename the columns using the dictionary
df1 = df1.rename(columns=new_column_names)
df2 = df2.rename(columns=new_column_names)

print(df1.head(10))
df1.dtypes


df1['Gross_Income'] = pd.to_numeric(df1['Gross_Income'], errors='coerce')

# Fill missing values with mean
df1.fillna(df1.mean(), inplace=True)

df2['Gross_Income'] = pd.to_numeric(df2['Gross_Income'], errors='coerce')

# Fill missing values with mean
df2.fillna(df2.mean(), inplace=True)

df1 = df1.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
df2 = df2.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

# Define the list of stop words
stop_words = set(stopwords.words('english'))

# Remove stop words from the 'text' column
df1['text'] = df1['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

df2['text'] = df2['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))



