# Udacity_Data_Science_Project_2 / # Disaster Response Pipeline Project
This is my repository for the second project of the data science course. It contains the following files:

### Files:
data/disaster_messages.csv -> Input file with message text, the columns are: id,message,original,genre

data/disaster_categories.csv -> Input file with massage categories, the columns are: id,categories

data/process_data.py -> Pyhon script to process the input data and store the result in a database. (Input: filepath to messages file, filepath to categories file, database filepath and name to be used; Output: Database with merged and cleaned data)

data/RobsDisasterResponse.db -> Output of process_data.py; Database with merged and cleaned data


models/train_classifier.py -> Python script to train a message clssifier and to store the model. (Input: filepath and name of database to be used, filepath for the trained model (name is hardcoded); Output: Prickle file with the trained model

models/robs_finalized_model.pkl -> Output of train_classifier.py; Pickle file with the trained model


app/run.py -> Python script to start a webserver that allows entering a message and displays the classification of the message according to the trained model. Warning: The paths and names for the database and the model are hardcoded here.

app/templates/go.html -> HTML file for displaying results

app/templates/master.html -> HTML file for entering messages

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model. Note that you need to use the same filepathes and names!

    - To run ETL pipeline that cleans data and stores in database:
     `python ./data/process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv data/RobsDisasterResponse.db`
    - To run ML pipeline that trains classifier and saves:
     `python .models/train_classifier.py /data/RobsDisasterResponse.db ./models`

2. Run the following command in the app's directory to run your web app:
     `python run.py`

3. Go to http://127.0.0.1:3001/

### Description
Processing the data: The ETL pipeline is pretty straight forward. The data is read from the two .csv files and merged into a pandas dataframe using the 'id'. The data is cleaned afterwards so that the categories become the titles of the columns and there are only integers in the classification. When looking at the data, it was found that there are some '2's in the categories. As we are looking at multi-label classification, these '2's were turned into '1's. Finally, duplicates (by id) were dropped. Finally, the data was saved into a database.

ML pipeline:
