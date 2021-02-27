# Installation
Jupyter Notebook `https://jupyter.org/install` is the only prerequisit to run the project.

# Project Overview
In this project, I analyze disaster data from Figure Eight `https://appen.com/` to build a model for an API that classifies disaster messages.
The data sets uded contain real messages that were sent during disaster events. A machine learning pipeline that categorizes
these events so that they could potentially be send to an appropriate disaster relief agency is employed.

# File Description

The notebook `ETL Pipeline Preparation.ipynb` contains a data cleaning pipeline that:
* Loads the `messages.csv` and `categories.csv` datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

The notebook `ML Pipeline Prepearation.ipynb` contains a Test processing and Machine Learning pipeline that:
* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set

# How to Interact with the Project
The Jupyter Notebooks contain markdowns and comments that guide the user into every step. 
As the data preparation is performed at the `ETL Pipeline Preparation.ipynb` one should run this notebook
first. `ML Pipeline Prepearation.ipynb` includes a Machine Learning pipeline that classifies the messages.






     
