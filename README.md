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
     `python ./models/train_classifier.py /data/RobsDisasterResponse.db ./models`

2. Run the following command in the app's directory to run your web app:
     `python run.py`

3. Go to http://127.0.0.1:3001/

### Description
Processing the data: The ETL pipeline is pretty straight forward. The data is read from the two .csv files and merged into a pandas dataframe using the 'id'. The data is cleaned afterwards so that the categories become the titles of the columns and there are only integers in the classification. When looking at the data, it was found that there are some '2's in the categories. As we are looking at multi-label classification, these '2's were turned into '1's. Finally, duplicates (by id) were dropped. Finally, the data was saved into a database.

ML pipeline: The ML pipeline first loads the data from the database. Then, category columns with only '0's are dropped. This is because they do not add any information to the training set and produce warnings in the later steps. In this dataset, the category 'child_alone' has only '0's and is therefore dropped. The dataset is then split into X (the messages) and y (the category labels) for the further steps. The data was quickly analyzed in terms of categories. The fgure below shows the distribution of labels:

![image](https://user-images.githubusercontent.com/65665840/119840848-9f1aaf80-bf05-11eb-95a7-b407646068e2.png)

It can be seen, that most of the data has no assiged label, many massees have one label assigned and there are some messages with more than 10 labels. The figure below shows the number of occurences for each category:

![image](https://user-images.githubusercontent.com/65665840/119841351-033d7380-bf06-11eb-866e-588a7c75f06f.png)

It has to be noted that the data is somewhat imbalanced as some labels occur only few times. This has an effect on the ML model as those lables are tended to be predicted to be 0 then.

After the data has been loaded, the data is split into a training and a test set using sklearn. Next, a learning pipeline is created. In consists of a CountVectorizer, a TfidfTransformer and a OneVsRestClassifier using a linear SVC estimator. In the CountVectorizer, a message is normalized (lower case, striped, urls removed), lemmatized and tokenized. Afterwads the TfidfTransformer computes the tf-idf for a given token. Finally, the token are classified OneVsRest with a linear SVC. 

This pipeline is used to fit the trining data. Afterwards the model is evaluated using the prediction of the X-test data and the true y_test data. Sci-Kit's classification report fnction is used to score the model. For the given setup, the following figure shows the results. An average f1 score of 0.69 was achieved. It can be seen that catgories with small sample numbers perform very poorly as indicated above.

![image](https://user-images.githubusercontent.com/65665840/119846284-4ac5fe80-bf0a-11eb-96f3-ba67c958f24b.png)

![image](https://user-images.githubusercontent.com/65665840/119907027-aebee600-bf4f-11eb-842c-a4efca14c92c.png)

![image](https://user-images.githubusercontent.com/65665840/119907087-d44bef80-bf4f-11eb-8189-740ff2c562e5.png)

In the next step, GridSearchCV is used to find improved parameters. Here, mainly the parameters for the CountVectorizer were alterd. Then the model was refitted with the improved parameters and evaluated again. The finalized model was then stored.

Visualization: Lastly, the data was visualized using the run.py script. A webserver is started that can be accessed at 127.0.0.1:3001. 
