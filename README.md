# Lending Club Predict

In this project, I am using data from [LendingClub.com](www.lendingclub.com), specifically from the years 2007-2010. I will be classifying anf predicting whether or not the borrower paid back their loan in full.

## Code and Resources Used

Python Version: Python 3.8 Packages Used: pandas, numpy, matplotlib, seaborn, sklearn

## Exploratory Data Analysis

I performed EDA across variables. Below are some highlights of my visualizations

add plot

addplot

addplot

## Building My Model

For this project, I created a single Decision Tree Model and a Random Forest Model to determine which model performs the best. Before doing so, I transformed the 'purpose' (categorical) column into a dummy variable. Then I split my data into training and test sets by a test size of 30%.

# Model Performance

After running my models, the Random Forest outperformed the Decision Tree. Although the Random Forest Model's accuracy was not 90% or above, it is still a decent measurement which can be used with caution. 

The Decision Tree scored an accuracy of 75%, whereas the Random Forest scored an accruacy of 84%.



