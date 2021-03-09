import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load Messages and Categories Data set from the csv file and merge  
    into a single data frame named df variable
    return a dataframe is merged on id column in both messages and categories data frame
    
    Input: messages_filepath, categories_filepath
    Output: Merged dataframe of messages and categories dataframe
    '''
   
    #Read csv file and load in the variable as dataframe
    messages =pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    
    # merge messages and categories on 'id' column
    df = pd.merge(messages, categories, on=["id"])
   
    
    return df
    
def clean_data(df):
    '''
    Function to clean the dataframe in order to be compatible for the ML application.
    Split the categories column with delimit ';' 
    Convert the first row values in categories dataframe to the column headers. 
    Replace the numerical values  2 to 1.
    Drop the duplicate rows from df dataframe.
    Drop the 'child_alone' column as it always takes the same value.
    Drop the column 'original' as it will not be used for the ML model.
    Drop rows with NA.
    Remove the existing 'categories' column from the df dataframe and concat the formatted 
    categories dataframe with df dataframe.   
    
    Input: df
    Output: cleaned and formatted dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";",expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    # removed everything after -
    row = row.str.split('-').str[0].tolist()
    # use this row to extract a list of new column names for categories.                
    category_colnames = row
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]  
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    #replace the 2 entries by 1
    categories.loc[(categories['related']==2)] = 1
    categories.drop(['child_alone'], axis=1, inplace=True)
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # drop original column as it is not needed for the ML model
    df.drop(['original'], axis=1, inplace=True)
    # drop rows with NA
    df.dropna(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    Function to save the cleaned dataframe into a sql database with file name 'disaster_clean'
    
    Input: df, database_filename
    Output: SQL Database 
    '''
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_clean', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()