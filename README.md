# How can we improve customers' retention ?

The objective of this project is to understand and predict the churn of a telecommunication firm customers, in order to improve customers' retention.

The data set comes from IBM Sample Data Sets
https://community.watsonanalytics.com/wp-content/uploads/2015/03/WA_Fn-UseC_-Telco-Customer-Churn.csv

# Introduction

The data set contains information on 7032 clients that subscrided a contract. Among those 7032 clients, 1869 clients have churned. The objective is to understand why they churned and provide a strategy to reduce this number.

##### available information : 

- Churn, Yes or No
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

##### Some visualisations : 

According to the graphs above, it looks like some variables such as the payment method or the type of contract have very different distributions among churners and non-churners : it's a good sign for the feasability of the project.


# Feature Engineering :

##### Rebalance the data : 

In this kind of classification problems, we need to work with balanced classes. Let's upsample the minority class:

```
minority=raw_data[raw_data.Churn=='Yes']
majority=raw_data[raw_data.Churn=='No']
df_minority_upsampled = resample(minority, 
                                 replace=True,     
                                 n_samples=len(majority),    
                                 random_state=123)
data_bal=pd.concat([df_minority_upsampled,majority])

```
