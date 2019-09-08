# Sentiment-Analysis-of-Amazon-Fine-Food-reviews
The analysis is to study Amazon food review from customers, and try to predict whether a review is positive or negative. The dataset contains more than 500K reviews with number of upvotes &amp; total votes to those comments.
### Learning Project
## 1.Business Problem
### 1.1 Description
#### Description
This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.

#### Problem Statemtent
Given a review, determine whether the review is positive (Rating of 4 or 5) or negative (rating of 1 or 2).

### 1.2 Source
https://www.kaggle.com/snap/amazon-fine-food-reviews

## 2.Machine learning problem
### 2.1 Data
#### 2.1.1 Data Overview
Number of reviews: 568,454<br>
Number of users: 256,059<br>
Number of products: 74,258<br>
Timespan: Oct 1999 - Oct 2012<br>
Number of Attributes/Columns in data: 10<br>

Attribute Information:<br>

1.Id<br>
2.ProductId - unique identifier for the product<br>
3.UserId - unqiue identifier for the user<br>
4.ProfileName<br>
5.HelpfulnessNumerator - number of users who found the review helpful<br>
6.HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not<br>
7.Score - rating between 1 and 5<br>
8.Time - timestamp for the review<br>
9.Summary - brief summary of the review<br>
10.Text - text of the review<br>

### 2.2 Mapping the real-world problem to a Machine Learning Problem
#### 2.2.1 Type of Machine Learning Problem
Binary Classification problem,Given a review we need to predict it is negative or postive reviews
#### 2.2.2 Performance metric
F1-score

## 3.Data Preprocessing
* Remove Special characters <br>
* Remove stop words<br>
* Remove HTML Tags<br>
* Removing null coloumn<br>
