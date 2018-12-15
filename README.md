The objective of this project is to understand and predict the churn of customers for a telecommunication company, in order to improve customers' retention. 

As the data is very imbalanced, I had to deal with imbalanced classification issues.
I used the imblearn package from Guilluame Lemaitre, in order to resample the majority and the minority class.

hypothesis : the telco company wants to identify all the churners. The action required when a customer is predicted as future churner is not expensive : as a consequence, FP are not an issue.

Here is a quick recap of the models' performances, (I will explain more in details when I have time) : 

```
                                  precision on majority     recall on minority  

logistic regression                 56 %                          85 %

logistic regression                 79 %                          90 %
(with penalization on minority)

undersampling with AllKNN           87 %                          93 %

undersampling AllKNN +              87 %                          93 %
SMOTE on minority
          
resampled + random forest           88 %                          93 %
          
```


