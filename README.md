### Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick_start)
2. [Project Overview](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The code should run using Python versions 3.*.
The necessary libraries are:

* pandas
* re
* sys
* json
* sklearn
* nltk
* sqlalchemy
* pickle
* Flask
* plotly
* sqlite3
* joblib


## Quick Start <a name="quick_start"></a>
Run the following commands in the project's root directory to set up database and model.
1. To run ETL pipeline that cleans data and stores in database
```
$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
2. To run ML pipeline that trains and saves the classifier 
```
$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
3. To run the web app go to `app` directory and execute: 
```
$ python run.py
```
4. To open the web app go to http://0.0.0.0:3001/

If you are only interested in running the web app go directly to steps 3 and 4.

## Project Overview <a name="motivation"></a>
In this project, I analyze disaster messages provided from Figure Eight `https://appen.com/` and build a web app that classifies messages
using a ML pipeline into 36 categories.  This classification could potentially help so that a message sent during a disaster could be effectively redirected
to the appropriate disaster relief agency. The dataset contains 30,000 messages taken from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010,
floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters. Some labels like aid-related, weather-related have much more entries compared to other categories, i.e. we have to deal with an imbalanced dataset. In the web app that is build with this training data one could
enter any disaster related message and a classification to related categories will be given to the user as output.


## File Descriptions <a name="files"></a>

<pre>
<code>.
├── <b>README.md</b>
├── <b>app</b> : Flask App Files
│ ├── <b>run.py</b> : Flask file to  run the app
│ └── <b>templates</b>
│ ├── <b>go.html</b>
│ └── <b>master.html</b>
├── <b>data</b> : It contains all ETL Files 
│ ├── <b>DisasterResponse.db</b> :  SQLite DataBase file containing cleaned data after ETL process
│ ├── <b>disaster_categories.csv</b> :  Disaster Categories CSV file
│ ├── <b>disaster_messages.csv</b> : Messages CSV file
│ └── <b>process_data.py</b> : ETL pipeline code
├── <b>models</b> : It contains all ML files
│ ├── <b>classifier.pkl</b> : classifier produced by train_classifier file
│ └── <b>train_classifier.py</b> : ML pipeline classification code
 </code>
</pre>

The file `process_data.py` contains an ETL pipeline that:
* Loads the `disaster_messages.csv` and `disaster_categories.csv` datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database `DisasterResponse.db`

The file `train_classifier.py` contains a NLP and ML pipeline that:
* Loads data from the SQLite database `DisasterResponse.db`
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Stores the model in `classifier.pkl` so that it can be used by the Flask app

The file `run.py` contains a Flask web app that enables the user to enter a disaster message, and then view the categories of the message.
 The web app also contains some visualizations that describe the data used to train the model.
 
## Results
When a disaster message is submitted and the Classify Message button is clicked, the app shows how the message is classified by highlighting the categories in green. For example for the message "We have nothing to eat!" we get the following categories: 'Request', 'Aid Related', 'Food'.
![Web app use picture](https://github.com/iris-theof/Disaster_response_pipeline/blob/master/screenshot_web_app.png)
 
## Licensing, Acknowledgements <a name="licensing"></a>
Must give credit to **figure-8** that provided the data used in this repository. Feel free to use the code in this repository as you would like! 








     
