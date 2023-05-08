# Toxicity Classsification Model

## 1. Problem Objective:
Online harassment and hate speech have become significant problems in the digital age, and there is a pressing need for automated tools that can detect and mitigate abusive behaviour. We are building a model that recognises toxicity and minimises bias with respect to mentions of identities. We will be using dataset labeled for identity mentions and optimising a metric designed to measure unintended bias. The reason for choosing this project is to create a tool that could help moderators in identifying and removing such comments, improve online discourse and create a safer environment for people to engage in online conversations.<br><br>
This project can contribute to creating a safer online environment by automating the process of identifying toxic comments and flagging them for human review. Additionally, the model could be further developed to handle various types of content in different languages, making it useful for a wide range of applications and conversations.

**Problem Statement:** Given a comment made by the user, predict the toxicity of the comment.

## 2. Our Approach:

### 2.1 Data: 

- Source: https://drive.google.com/drive/folders/1WMY6VMZ81LD2oblBI1MVtAup4M2I8xX7?usp=share_link
<br><i>(Due to large size of csv files, we have to use external resource to host them)</i><br><br>
- We have one single csv file for training and one cvs file to test.
- Columns in train data:
	- Comment_text: This is the data in string format which we have to use to find the toxicity.
	- target: Target values which are to be predicted (has values between 0 and 1)
	- Data also has additional toxicity subtype attributes: (Model does not have to predict these)
		- severe_toxicity
		- obscene
		- threat
		- insult
		- identity_attack
		- sexual_explicit
	- Comment_text data also has identity attributes carved out from it, some of which are:
		- male
		- female
		- homosexual_gay_or_lesbian
		- christian
		- jewish
	        - muslim
		- black
		- white
		- asian
		- latino
		- psychiatric_or_mental_illness
	- Apart from above features the train data also provides meta-data from jigsaw like:
		- toxicity_annotator_count
		- identity_anotator_count
		- article_id
		- funny
		- sad
		- wow
		- likes
		- disagree
		- publication_id
		- parent_id
		- article_id
		- created_date

### 2.2 Type of Machine Learning Problem:
We have to predict the toxicity level(target attribute). The values range from 0 to 1 inclusive. This is a regression problem. It can also be treated as a classification problem if we take every value below 0.5 to be non-toxic and above it to be toxic, we would then get a binary classification problem.



### 2.4 Performance Metric:
For our training and evaluation we will use the MSE(Mean Squared Error).

### 2.5 Machine Learning Objectives and Constraints:

**Objectives:** Predict the toxicity of a comment made by the user. (0 -> not toxic, 1 -> highest toxicity level)

**Constraints:**

- The model should be fast to predict the toxicity rating.
- Interpretability is not needed.

## Team Members

- <a href="https://sharmahemang.com/">Hemang Sharma (UTS ID: 24695785)</a>
- <a href="https://canvas.uts.edu.au/groups/157259/users/132836">Nusrat Zahan (UTS ID: 14367472)</a>
- <a href="https://canvas.uts.edu.au/groups/157259/users/132900">Rajveer Singh Saini (UTS ID: 14368005)</a>

## Refrences 
1. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
2. https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/
3. https://www.geeksforgeeks.org/decision-tree/
4. https://www.geeksforgeeks.org/decision-tree-implementation-python/
5. https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
