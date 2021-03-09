import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


def load_data(database_filepath):
    
    '''
    Function to retreive data from sql database (database_filepath) and split the dataframe into X and y variable
    
    Input: Databased filepath
    Output: Returns the Features X & target y along with target columns names catgeory_names
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('disaster_clean',con=engine)
    #  define feature and target variables X and Y
    X = df['message'].values 
    y = df[df.columns[4:]]
    category_names = y.columns.tolist()
    
    return X, y, category_names


def tokenize(text):
    '''
    Function that splits text into words and return the root form of the words
    after removing the stop words
    
    Input: text(str): the message
    Output: lemm(list of str): a list of the root form of the message words
    '''
    #Regex to find urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Finds all urls from the provided text
    detected_urls = re.findall(url_regex, text)
    
    #Replaces all urls found with the "urlplaceholder"
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    
        
    # Extracts the word tokens from the provided text    
    tokens = word_tokenize(text)
      
    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in tokens if t not in stop]
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = WordNetLemmatizer()

    # Makes a list of clean tokens
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Function that specifies the pipeline and
    the grid search parameters in order to build a
    classification model
     
    Output:  cv: classification model
    '''
    
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(GradientBoostingClassifier(max_depth=6)))
    ])
    
    # choose parameters
    parameters = {'clf__estimator__n_estimators': [100, 140]}

    # create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='recall_micro', cv=4)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate the model printing scores for each category_name
    Input: model, X_test, Y_test, category_names
    Output: classification report and accuracy score for all category_names
    '''
    
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))
    


def save_model(model, model_filepath):
    '''
    Function to save a pickle file of the model
    
    Input:
    model: the classification model
    model_filepath (str): the path of pickle file
    '''

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
