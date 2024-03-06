Movie Review Sentiment Analysis

Overview
This project performs sentiment analysis on movie reviews extracted from the IMDb dataset. It aims to classify movie reviews as positive or negative based on their sentiment.

Dataset
The IMDb dataset contains a large collection of movie reviews along with their corresponding sentiment labels. For this project, a subset of 9000 positive reviews and 1000 negative reviews was selected to create a balanced dataset for training and testing.

Dependencies
•	Python 3.x
•	Libraries:
  •	NumPy
  •	Pandas
  •	Matplotlib
  •	Seaborn
  •	scikit-learn
  •	imbalanced-learn

Model Selection
The project trains several machine learning models including:
  •	Support Vector Machine (SVM) with a linear kernel
  •	Decision Tree
  •	Naive Bayes
  •	Logistic Regression
After evaluating the models on the test set, the SVM with a linear kernel was selected as the best-performing model based on accuracy and F1 score.

Hyperparameter Tuning
Hyperparameters for the SVM model were tuned using GridSearchCV. The best parameters obtained were C = 1 and kernel = 'linear'.

Future Work
•	Explore deep learning models such as LSTM or Transformer-based architectures for sentiment analysis.
•	Experiment with different preprocessing techniques and feature engineering approaches.
•	Incorporate word embeddings like Word2Vec or GloVe for better representation of text data.

