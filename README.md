# How can we improve customers' retention ?

The objective of this project is to understand and predict the churn of a telecommunication firm customers, in order to improve customers' retention.



# Table of Contents

* [Introduction](# Introduction)
* [Feature Engineering](# Feature Engineering) 
* [Machine learning modeling](# Machine learning modeling)
* [Interpretation of the model](# Interpretation of the model)
* [Conclusions](# Conclusions)
* [What's next?](# What's next?)


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


# Feature Engineering 


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



# Machine learning modeling



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


# Interpretation of the model


#### visualization

Random forests rely on two principles :

- bagging : choose B samples and train decision trees on each of these B samples. Then we take the mode of all the predictions in order to determine the predicted class.
- random subset of features : each decision tree is trained on a subset of features

Random forests are often considered as a "black box" but some tools can make their interpretation quite straightforward.
For instance, I decided to use graphviz, a visualization tool for decision trees, in order to plot one of the M decision trees of our random forest. Here I choose trees of small maximum depth (=3) because deep trees are impossible to display:






![](RF_inter_3.png?raw=false)







We can see in the upper cell that we initially have 7228 individuals in our training set. The ```value``` array indicates the population is divided in two classes: 'no churn' (3604) and 'churn' (3624). We can read ```class = Churn ``` because 'churn' is the dominant class (3624 > 3604).

The first decision is based on the value of 'Contract_Two year' : 
  
  If ```(Contract_Two year <= 0.5) == False ``` , i.e  ```Contract = Two year ``` , then we go right. Among the initial 7228   individuals, only 1206 have a two year contract. Among those 1206 individuals, 1114 belong to the 'no churn' class. As a consequence, people with a two year contract are strongly likely to do not churn. We can make similar analyses for the other branches of the tree.
  
  
  

# Conclusions

# What's next ?



