import joblib
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from wordcloud import WordCloud
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly
from flask import Flask, render_template, url_for, request

# numpy and pandas for data manipulation
import pandas as pd
import numpy as np
from numpy import median
from scipy.stats import norm
import re
import math

# matplotlib and seaborn for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set(style='darkgrid')


# file system management


seed = 42


columns = ['name', 'item_condition_id','category_name', 'brand_name','shipping', 'item_description']
app = Flask(__name__)


NUM_BRANDS = 2500
# NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        features = [x for x in request.form.values()]
        data = np.array(features)
        df_pred = pd.DataFrame([data], columns=columns)
        print("----------df_pred------------")
        print(df_pred.dtypes)
        

        df_train = pd.read_csv('../Solutions/train.csv')
        df_test = pd.read_csv('../Solutions/test.csv')

        nrow_train1 = df_train.shape[0]

        df_t = pd.concat([df_test,df_pred])
        df_t.iloc[-1, df_t.columns.get_loc('id')] = 321123
        df_t.iloc[-1, df_t.columns.get_loc('seller_id')] = 4353434
        df_t.to_csv(r'../t.csv', index = False)
        df_t = pd.read_csv('../t.csv')
        os.remove('../t.csv')
        df = pd.concat([df_train, df_t], 0)

        print("----------df-----------")
        print("last row after")
        print(df.tail())
        print(df.tail(1).dtypes)

        brands = df[:nrow_train1].groupby('brand_name')['price'].agg(['count', 'mean']).sort_values(by=['count'], ascending=False).reset_index()

        luxurious_brands = brands[:20]
        
        
        brands = df[:nrow_train1].groupby('brand_name')['price'].agg(['count', 'mean']).sort_values(by=['count'], ascending=False).reset_index()
        
        cheap_brands = brands[:10]
        
        
        brands = df[:nrow_train1].groupby('brand_name')['price'].agg(['count', 'mean']).sort_values(by=['count'], ascending=False).reset_index()
        
        expensive_brands = brands[:20]
        
        
        #stopwords without no, not, etc
        STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
                    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
                    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
                    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
                    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
                    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
                    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
                    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
                    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
                    'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
                    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
                    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
                    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
                    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
                    'won', "won't", 'wouldn', "wouldn't"]
        
        
        def remove_emoji(sentence):
            pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
            
            return pattern.sub(r'', sentence)
        
        def process_category(input_data):
            for i in range(3):
                
                def get_categories(ele):
                    
                    if type(ele) != str:
                        return np.nan
                
                    cat = ele.split('/')
                    
                    if i >= len(cat):
                        return np.nan
                    else:
                        return cat[i]
        
                col_name = 'category_' + str(i)
                
                input_data[col_name] = input_data['category_name'].apply(get_categories)
                
                input_data.fillna({'category_name': 'Other'}, inplace = True)
            
            return input_data
        
        def decontracted(phrase):
            phrase = re.sub(r"won't", "will not", phrase)
            phrase = re.sub(r"can\'t", "can not", phrase)
            phrase = re.sub(r"n\'t", " not", phrase)
            phrase = re.sub(r"\'re", " are", phrase)
            phrase = re.sub(r"\'s", " is", phrase)
            phrase = re.sub(r"\'d", " would", phrase)
            phrase = re.sub(r"\'ll", " will", phrase)
            phrase = re.sub(r"\'t", " not", phrase)
            phrase = re.sub(r"\'ve", " have", phrase)
            phrase = re.sub(r"\'m", " am", phrase)
            
            return phrase

        def process_text(input_data, cols):
            for col in cols:
                
                processed_data = []
                
                for sent in input_data[col].values:
                    
                    sent = decontracted(sent)
                    sent = sent.replace('\\r', ' ')
                    sent = sent.replace('\\"', ' ')
                    sent = sent.replace('\\n', ' ')
                    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
                    sent = remove_emoji(sent)
                    sent = ' '.join(e for e in sent.split() if e not in STOPWORDS)
                    processed_data.append(sent.lower().strip())
                    
                input_data[col] = processed_data
                
            return input_data
        
        def handle_missing_values(input_data):
            input_data.fillna({'name': 'missing', 'item_description': 'missing'}, inplace=True)
            
            return input_data
        
        
        #nlp features
        def get_text_features(input_data):
            input_data['is_luxurious'] = (input_data['brand_name'].isin(luxurious_brands['brand_name'])).astype(np.int8)
        
            input_data['is_expensive'] = (input_data['brand_name'].isin(expensive_brands['brand_name'])).astype(np.int8)
        
            input_data['is_cheap'] = (input_data['brand_name'].isin(cheap_brands['brand_name'])).astype(np.int8)
        
            return input_data
        
        
        def preprocess(input_data):
            input_data['price'] = np.log1p(input_data['price'])
        
            input_data = handle_missing_values(input_data)
            
            input_data = process_category(input_data)
            
            input_data = process_text(input_data, ['name', 'item_description', 'category_name'])
        
            return input_data

        data = preprocess(df)
        data.fillna({'category_0': 'other', 'category_1': 'other', 'category_2': 'other'}, inplace = True)
        
        #NLP features
        data = get_text_features(data)
        
        data.fillna({'brand_name': ' '}, inplace = True)
        
        #concatenate text features
        data['name'] = data['name'] + ' ' + data['brand_name'] + ' ' + data['category_name']
        data['text'] = data['name'] + ' ' + data['item_description']
        
        data = data.drop(columns = ['brand_name', 'item_description', 'category_name'], axis = 1) 
        data = data[['price', 'name', 'category_0', 'category_1',
               'category_2', 'shipping', 'item_condition_id', 'is_expensive', 'is_luxurious', 'text']]
        
        #one hot encoding of category names
        def get_ohe(X_train, col_name):
            vect = CountVectorizer()
            tr_ohe = vect.fit_transform(X_train[col_name].values)
            return tr_ohe
        
        #tfidf word embeddings
        def get_text_encodings(X_train, col_name, min_val, max_val):
            vect = TfidfVectorizer(min_df = 10, ngram_range = (min_val, max_val), max_features = 1000000)
            tr_text = vect.fit_transform(X_train[col_name].values)
            return tr_text
        
        def generate_encodings(X_train):
            tr_ohe_category_0 = get_ohe(X_train, 'category_0')
            tr_ohe_category_1 = get_ohe(X_train,'category_1')
            tr_ohe_category_2 = get_ohe(X_train,'category_2')

            
            tr_trans = csr_matrix(pd.get_dummies(X_train[['shipping', 'item_condition_id', 'is_expensive', 'is_luxurious']], sparse=True).values)
            
            tr_name = get_text_encodings(X_train, 'name', 1, 1)
            tr_text = get_text_encodings(X_train, 'text', 1, 2)
        
            train_data = hstack((tr_ohe_category_0, tr_ohe_category_1, tr_ohe_category_2, tr_trans,
                               tr_name, tr_text)).tocsr().astype('float32')
        
            return train_data
        y = data['price']
        
        X = data.drop('price', axis = 1)

        X = generate_encodings(X)
        print(X)
        X_train = X[:nrow_train1]
        Y_train = y[:nrow_train1]
        X_test = X[nrow_train1:]
        print(X_train)
        print("shape of X_train", X_train.shape)
        #ridge
        ridge_model = Ridge(solver='lsqr', fit_intercept=False) #solver='lsqr' reduces time to train significantly
        ridge_model.fit(X_train, Y_train)
        test_pred = np.expm1(ridge_model.predict(X_test))
        print(test_pred)
        my_prediction = math.ceil(test_pred[-1])
    
    return render_template('home.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
