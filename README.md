The objective of this project is to understand and predict the churn of customers for a telecommunication company, in order to improve customers' retention. 
We will try to have a business oriented explanation of the project, so please go to the [github repo](https://github.com/guillaumedelaloy/Churn_analysis) for more details.



# Table of Contents

* [Introduction](#introduction)
  + [Data overview](#data-overview)
  + [Some visualizations ](#some-visualizations)
* [Feature Engineering](#feature-engineering)
  + [Rebalance the data](rebalance-the-data)
  + [Dealing with colinearity](#dealing-with-colinearity)
* [Machine learning modeling](#machine-learning-modeling)
  + [Logistic Regression](#logistic-regression)
  + [Random Forest](#random-forest)
* [Interpretation of the model](#interpretation-of-the-model)
  + [A unified interpretation](#a-unified-interpretation)
* [Conclusions](#conclusions)
* [Next steps](#next-steps)
* [Weaknesses of this dataset](#weaknesses-of-this-dataset)


## Introduction

#### Data overview

The data set comes from [IBM Sample Data Sets](https://community.watsonanalytics.com/wp-content/uploads/2015/03/WA_Fn-UseC_-Telco-Customer-Churn.csv)

The data set contains information on 7032 clients who subscribed a contract. Among those 7032 clients, 1869 clients have churned. The objective is to understand why they churned and provide a strategy to reduce this number.

- Churn, Yes or No
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

#### Some visualizations 


Many churners have a month to month subscription :


<p align="center">
  <img src= "https://github.com/guillaumedelaloy/Churn_analysis/blob/master/graphs/Churn_Contract.png?raw=true">
</p>





Clients with Fiber optic have a high churn rate :


<p align="center">
  <img src= "https://github.com/guillaumedelaloy/Churn_analysis/blob/master/graphs/Internet_churn.png?raw=true">
</p>



According to the graphs above, some variables such as the internet service or the type of contract have very different distributions among churners and non-churners : it's a good sign for the feasability of the project.


## Feature Engineering 


#### Rebalance the data 

In this kind of classification problems, we need to work with balanced classes. Let's upsample the minority class :


to do : update work with the imblearn library to undersample the majority class with Knn


#### Dealing with colinearity 


We first convert the categorical variables to dummies :
```
data=pd.get_dummies(data_bal[categorical_var],drop_first=True)
```
The parameter drop_first=True means that one categorical variable taking m possible values will be converted to m-1 dummies (and not m dummies).

Now let's check the correlations between the 32 variables :



<p align="center">
  <img src= "https://github.com/guillaumedelaloy/Churn_analysis/blob/master/graphs/Churn_corr.png?raw=true">
</p>


Many variables are very correlated. For instance, 'InternetService_No' is obviously highly correlated with 'OnlineSecurity_No internet service'. 

In order to remove the colinearities, we implement the following idea : 

for each variable, we look for the variables with a correlation above a threshold. Then we order those variables by descending correlation and we add the first one to a list. At the end of the while loop, the list contains all the variables we will remove from the explanatory variables. After some tests, we choose 0.85 as the optimal threshold and we obtain the following correlations:


<p align="center">
  <img src= "https://github.com/guillaumedelaloy/Churn_analysis/blob/master/graphs/Churn_decorr.png?raw=true">
</p>




## Machine learning modeling



goal 1 : detect efficiently the customers who are likely to churn. We assume that the company would like to be able to target all the churners among the smallest population of customers. In other words, if the model predicts 'Churner' for a customer, we want to be sure that it is the case. In order to fulfill this objective, we would focus on the recall for the churners' class.

goal 2 : Have a model with a good interpretability, in order to be able to prevent the events that lead to a churn


#### Logistic Regression


Logistic regressions are very good for interpretabilty but usually not so acurrate. We implement a quick grid search in order to optimize the parameters and obtain the following results with the best parameters :

```
              precision    recall  f1-score   support

          0       0.79      0.75      0.77      1559
          1       0.76      0.79      0.78      1539

avg / total       0.77      0.77      0.77      3098
```


#### Random Forest


We obtain better results with a random forest model. It is particularly interesting to see that the recall for class 1, ie the 'churners' is 93% : 


```
              precision    recall  f1-score   support

          0       0.92      0.86      0.89      1559
          1       0.87      0.93      0.90      1539

avg / total       0.90      0.90      0.90      3098
```

We could have slightly better results with an SVM classifier (see code [here](https://github.com/guillaumedelaloy/Churn_analysis/blob/master/churn_prediction.ipynb)).
However, SVMs are really hard to interpret so we won't choose this model.


## Interpretation of the model

#### A unified interpretation

While scrolling my Linkedin feed, I found a medium article related to machine learning model interpretation and a new package called [shap](https://github.com/slundberg/shap), inspired from a 2017 [paper](https://arxiv.org/abs/1705.07874). Here is the result I obtained with my RF model:


<p align="center">
  <img src= "https://github.com/guillaumedelaloy/Churn_analysis/blob/master/graphs/Shap_interpretation.png?raw=true">
</p>


To conclude, tenure has a strong negative impact on churn probability when taking high values(red points), while it has a strong positive impact when taking low values(blue points), ie it increases the churn probability. For binary factors, low values means 0/False and high values means 1/True.




  

## Conclusions

1: We can predict with good performances the churn of customers

2: The tenure, contract, Internet service, type of payment and monthly charges are the 5 most important factors to determine the churn.

3: In average, the categories ```tenure < 24```, ```month to month contract```, ```fiber optic```, ```paying by electronic check```, with ```monthly charges > 60``` increase the churn probability


## Next Steps

Now if your company needs to actually tackle a churn problem, here is the process you need to implement:

1: Train and deploy periodically the model we chose on your customers' database(s)

2: Compute the churn probability given by your model for each customer, and write the result in the database

3: Then make this information available to the people interacting with the client, so that they can focus more on the clients with high churn probabilities.

## Weaknesses of this dataset

The overall quality of the data is good but here are some ways of improvement:

1: The size of the data is 7K lines, which is rather small. Bigger size will improve the trust in the results.

2: No temporal data while I assume a churn can be explained by several past events



