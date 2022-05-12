# Kaggle Challenge

# Task:

To Use Machine Learning to Predict Price of Product listed and get rmsle < 0.5 and Create a web app to take inputs and display the predicted price.

# Directory structure:
<pre>
m-rec/8e23ae047060859fed073e78fb70e4c2f5160b57/
├── README.md             # overview of the project
├── Deployment/           # contains code for Flask Web Application
    ├── app.py            # contain the main code for running the web app
    ├── requirements.txt  # software requirements and dependencies
├── Solutions/            # Contains all the Models used for prediction in this project
    ├── EDA.ipynb         # Contains all the codes for Exploratory data analysis of the dataset (train) provided
    ├── model.ipynb       # Contains the main code for preprocessing and price prediction using ML model (rmsle : 4.9)
    ├── rough/            # Contains all the rough codes that I tried to test on the dataset
├── requirements.txt      # software requirements and dependencies
├── data/                 # Contains the dataset (Train + Test Files)
├── submission.csv        # Contains the predicted values for all the products given in ./data/mercari_test.csv.gz
</pre>

# Process Involved: 

I followed this Strategy to approach the solution to this problem.

1. Target definition (Given)
2. Data collection & preparation (Given)
3. Feature engineering
4. Model training
5. Model evaluation
6. Model deployment (Used Flask-a python based web framework)


# Technologies Used:


<div align="center">
<code><img height="100" src="https://camo.githubusercontent.com/fc4cab9ccd5e6e62ac62dbb5aab11a9e5507b438c42cc82363ce184cbe1ccdaa/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f632f63332f507974686f6e2d6c6f676f2d6e6f746578742e7376672f3230303070782d507974686f6e2d6c6f676f2d6e6f746578742e7376672e706e67" /></code>
<code><img height="80" src="https://www.kdnuggets.com/wp-content/uploads/jupyter-logo.jpg" /></code>
<code><img height="80" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" /></code>
 </div>

# Instructions:

* Clone the repository in your local pc.
* Make sure you are in the root directory.
* Run the following code:
```python:
pip install -r requirements.txt                (Python 2)
pip3 install -r requirements.txt               (Python 3)
```
* To test the model type the code below and goto to ./Solutions/model.ipynb and run the jupyter notebook.
```python:
cd Solutions
```
* To test the flask website change directory to Deployment, then install the requirements.txt, and then run app.py using following code:
```python:
cd Deployment
pip3 install -r requirements.txt               (Python 3)
python app.py runserver
```
(If It doesn't work on the first go , try refreshing the website and try again)
* Type the inputs present in the website like this :

![mercari](https://user-images.githubusercontent.com/58468853/147400316-31be863c-31d2-45fe-9a66-4bc40749ba45.png)

* The Predicted price Will be displayed in few seconds , like this :

![mercari_pred](https://user-images.githubusercontent.com/58468853/147400332-07a2df5f-2abb-496d-9144-bf7bf3f478c8.png)

# Observations and Results:
* Ridge regression with best hyperparameters takes very less time to train and rmsle is also less than 0.5, so I choose ridge_model to predict the outcome of test_data or mercari_test. Got rmsle as 0.49 using it.
* The predicted values for products given in ./data/mercari_test.csv.gz is present in submission.csv
* A full stack website built with Flask as backend and HTML, CSS for frontend is ready to predict the price of products based on input parameters given.

