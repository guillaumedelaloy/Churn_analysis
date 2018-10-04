# How can we improve customers' retention ?

The objective of this project is to understand and predict the churn of a telecommunication firm customers, in order to improve customers' retention. 
Here I will have a business oriented explanation of the project, please go on the [github repo](https://github.com/guillaumedelaloy/Churn_analysis) for more details.



# Table of Contents

* [Introduction](#introduction)
* [Feature Engineering](#feature-Engineering) 
* [Machine learning modeling](#machine-learning-modeling)
* [Interpretation of the model](#interpretation-of-the-model)
* [Conclusions](#conclusions)
* [What's next?](#what's-next?)
* [Weaknesses of this dataset](#weaknesses-of-this-dataset)


## Introduction

#### Data overview

The data set comes from [IBM Sample Data Sets](https://community.watsonanalytics.com/wp-content/uploads/2015/03/WA_Fn-UseC_-Telco-Customer-Churn.csv)

The data set contains information on 7032 clients that subscrided a contract. Among those 7032 clients, 1869 clients have churned. The objective is to understand why they churned and provide a strategy to reduce this number.

- Churn, Yes or No
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

#### Some visualisations 


Many churners have a month to month subscription :



![](Churn_Contract.png?raw=false )




Clients with Fiber optic have a high churn rate :



![](Internet_churn.png?raw=true)



According to the graphs above, it looks like some variables such as the internet service or the type of contract have very different distributions among churners and non-churners : it's a good sign for the feasability of the project.


## Feature Engineering 


#### Rebalance the data 

In this kind of classification problems, we need to work with balanced classes. Let's upsample the minority class :

```
minority=raw_data[raw_data.Churn=='Yes']
majority=raw_data[raw_data.Churn=='No']
df_minority_upsampled = resample(minority, 
                                 replace=True,     
                                 n_samples=len(majority),    
                                 random_state=123)
data_bal=pd.concat([df_minority_upsampled,majority])

```


#### Dealing with colinearity 


We first convert the categorical variables to dummies :
```
data=pd.get_dummies(data_bal[categorical_var],drop_first=True)
```
The parameter drop_first=True means that one categorical variable taking m possible values will be converted to m-1 dummies (and not m dummies).

Now let's check the correlations between the 32 variables :



![](Churn_corr.png?raw=false )



We can see that many variables are very correlated. For instance, 'InternetService_No' is obviously highly correlated with 'OnlineSecurity_No internet service'. 

In order to remove the colinearities, we implement the following idea : 

for each variable, we look for the variables with a correlation above a threshold. Then we order those variables by descending correlation and we add the first one to a list. At the end of the while loop, the list contains all the variables we will remove from the explanatory variables. After some tests, we choose 0.85 as the optimal threshold and we obtain the following correlations:



![](Churn_decorr.png?raw=false)



## Machine learning modeling



We have two goals here :

goal 1 : detect efficiently the customers that are likely to churn. We assume that the company would like to be able to target all the churners among the smallest population of customers. In other words, if the model predicts 'Churner' for a customer, we want to be sure that it is the case. In order to fulfill this objective, we would focus on the recall for the churners' class.

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


#### Decision tree

Random forests rely on two principles :

- bagging : choose B samples and train decision trees on each of these B samples. Then we take the mode of all the predictions in order to determine the predicted class.
- random subset of features : each decision tree is trained on a subset of features

Random forests are often considered as a "black box" but some tools can make their interpretation quite straightforward.
For instance, I decided to use graphviz, a visualization tool for decision trees, in order to plot one of the M decision trees of our random forest. Here I choose a tree of small maximum depth (=3) because deep trees are impossible to display:





![](RF_inter_3.png?raw=false)







We can see in the upper cell that we initially have 7228 individuals in our training set. The ```value``` array indicates the population is divided in two classes: 'no churn' (3604) and 'churn' (3624). We can read ```class = Churn ``` because 'churn' is the dominant class (3624 > 3604).

The first decision is based on the value of 'Contract_Two year' : 
  
  If ```(Contract_Two year <= 0.5) == False ``` , i.e  ```Contract = Two year ``` , then we go right. Among the initial 7228   individuals, only 1206 have a two year contract. Among those 1206 individuals, 1114 belong to the 'no churn' class. As a consequence, people with a two year contract are strongly likely to do not churn. We can make similar analyses for the other branches of the tree.
    



#### Feature importance





Let's now have a look at the importance of each feature in the model :




![](feature_importance.png?raw=false)





We can see that ``` ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'PaymentMethod'] ``` contributed to 70% of the predictive power of our model. I summed the feature importances of the dummies under their corresponding categorical variables for interpretation purposes.



#### Feature impact on churn probability

Now that we know what features are important, let's investigate how each feature impacts the churn probability.
In order to do this, i used the  ``` treeinterpreter ``` package. Here is the example of three customers. Each customer has an initial probability of churn of 0.5. Then, depending on the values of each feature we add/substract the feature's contribution (written in parenthesis). 






![](impact_proba.png?raw=true)






Let's have an example to better understand what a contribution means:

For customer 1, the tenure is 9, which means it's been 9 months the client subscribed the company's offer, and has a contribution of 0.08. That is to say that, for this customer, a tenure of 9 makes the customer more likely to churn. 
For customer 2, the tenure is also 9 but the contribution is -0.03, which means the tenure contributes to slighlty lower the likelihood of churn. Why? Because customer 1 and 2 have different characteristics. For instance, one has a one year contract while the other has a month to month contract. Since, customer 1 has a one year contract, it means that he will have to decide or not to subscribe again to the offer in 3 months. While a month to month contract can be stopped at the end of each month.
We can correlate this intuition with the following graph:





![](tenure_contrib_evo.png?raw=true)






We can clearly see that, in average, the tenure contribution for the month to month contracts decreases faster than for the one year and two year contract. Moreover, an interesting insight is that we can see all three regression lines going below zero when tenure goes above 24 (two years). This means that when tenure is above 24, the contribution becomes negative, i.e tenure > 24 decreases the churn probability. When tenure <=24, the contribution is positive and increases the churn probability.

We can have similar analysis with the other variables. We obtain that the "breakeven point" for MonthlyCharges is 60 : a contract more expensive than 60$ per month increases the churn probability.






![](contrib_monthly_evo.png?raw=true)






  

## Conclusions

1: We can predict with good performances the churn of customers

2: The tenure, contract, Internet service, type of payment and monthly charges are the 5 most important factors to determine the churn.

3: We know that, in average, the categories ```tenure < 24```, ```month to month contract```, ```fiber optic```, ```paying by electronic check```, with ```monthly charges > 60``` increase the churn probability


## What's next ?

Now if your company needs to actually tackle a churn problem, here is the process you need to implement:

1: Train and deploy periodically the model we chose on your customers' database(s)

2: Compute the churn probability given by your model for each customer, and write the result in the database

3: Then make this information available to the people interacting with the client, so that they can focus more on the clients with high churn probabilities.

## Weaknesses of this dataset

The overall quality of the data is good but here are some ways of improvement:

1: The size of the data is 7K lines, which is rather small. Bigger size will improve the trust in the results.

2: We don't have temporal data while I assume a churn can be explained by several past events



