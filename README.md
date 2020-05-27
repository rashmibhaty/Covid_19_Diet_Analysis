# Covid_19_Diet_Analysis
Different machine learning techniques are used Analyse Covid19 and Diet plan.
------------------------------------------------------------------------
Analysis based on Covid-19 Confirmed cases and Protein Intake 
------------------------------------------------------------------------
To understand the relationship between Covid-19 and Source of Protein, I have applied Principal Component Analysis(PCA) to the data
from Protein_Supply_Quantity_Data.csv. Dataset provides us the % of Protein Source (i.e. From Animal Products, Milk, Pulses etc.,) for
different countries and confirmed COVID-19 cases.

Using PCA Technique, I have tried to analyse and find out the most important factor of Protein Source.

It is found out that Animal Products are the most important factor( Principal Component) with explained_varianceratio of 0.25363972.

It is also found that "Vegetable Products" column is redundant.If we remove "Animal Products" feature from analysis then 
"Vegetable Products" becomes principle component.

I have plotted Principal Component Vs Confirmed Cases percentage to see relationship between them.
We can see that COVID-19 confirmed percentage is less in the contries where Animal Product Protein consumption is low like India.

Amongst the contries where Animal Product Protein consumption is high, we see confirmed percentage also high,like Spain, 
United States of America, Italy.

However, there are countries like Australia where confirmed percentage is low despite High % of Protein intake from 
Animal Product source, may be, perhaps, due to other fitness habits and wide spread population etc.
--------------------------------------------------------------
------------------------------------------------------------------
Analysis based on Confirmed Cases Percentage and Obesity Percentage
------------------------------------------------------------------
To understand the relationship between Covid-19 and Obesity, I have applied K Means clustering to the data obtained
from Fat_Supply_Quantity_Data.csv.
Used Elbow method to find optimum number of clusters. In this case, it was found that 3 is the optimum number of clusters.

Cluster 0 -- Low COVID-19 confirmed percentage and Low Obesity percentage.
      Ex. India,Kenya, Thailand
Cluster 1 -- Low COVID-19 confirmed percentage and High Obesity percentage.
      Ex. Australia, Jordan, Mexico
Cluster 2 -- High COVID-19 confirmed percentage and High Obesity percentage.
      Ex. United States of America, Spain, Iceland             
So we can see that contries where obesity is low COVID-19 cases are low.
-------------------------------------------------------------------------------
