# Reccomender Systems 

## Matrix Factorization Machine and Youtube Candidate Generator Model
---
The project is based on the following papers:
* [Matrix Factorization](https://arxiv.org/pdf/2203.11026.pdf)
* [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
---
Dataset:
* [MovieLens 100k](https://www.kaggle.com/datasets/ayushimishra2809/movielens-dataset)
---
### Results
#### Model Architectures
Factorization Machine            |  Candidate Generator Model
:-------------------------:|:-------------------------:
![](https://github.com/lukabarbakadze/Youtube-RecSys/blob/main/charts/FactorizationMatrix.png)  |  ![](https://github.com/lukabarbakadze/Youtube-RecSys/blob/main/charts/CandidateGenerator.png)
#### Matrix Factorization Model
- Strengths:
* * Computationally inexpensive model
* * Can handle sparse data
* * Quite accurately predicts expected rating
- Weakneses:
* * Only capturaes linear relationship between independent and dependent variable
* * Limited number of users/products (if we want to add new user/item in the system, we have to assign correponding user/movie embedding to it by retraining the model)
* * Easy to overfit the training data
#### Youtube Candidate Generation Model
- Strengths:
* * Do not have user's limit (user's embedding depends only his/her history)
* * Captures non-linear and relatively complex relationships between user and products
- Weaknesses
* * Computationally expensive (especially when there is high number of products)
* * Requires more amount of data and more feature engineering
* * Still limited to number of products (because of pre-trained embeddings, adding new product will requre to assign same dimensional embedding vector to new product too)
#### Model Training
![1](https://github.com/lukabarbakadze/Youtube-RecSys/blob/main/charts/training.png)

### Files Description
* models.py - Pytorch (from scratch) implementation of Matrix Factorization and Youtube Candidate Generator models
* main.ipynb - Main working&training file
* requirements.txt - Dependencies used in the project
---
### Table of Contents
* Imports
* Data Preprocessing
* * Feature Engineering
* Matrix Factorization
* Youtube Candidate Generator model
* * Feature Engineering II
* Vizualization
---
