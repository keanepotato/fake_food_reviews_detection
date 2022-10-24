# Instead of Random forests, use a different type of decision trees
#%%
#from sklearn.externals import joblib
import joblib
import gradio as gr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

# def stem_words(text):
#     return ' '.join([stemmer.stem(word) for word in text.split()])

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


#%% PREPROCESSING OF TEXT 
preprocess_req = False
if preprocess_req:
    df = pd.read_csv("shortened_Yelp_Food_Reviews_Dataset.csv")
    len(df)
    #stemmer = PorterStemmer()
    
    # Removal of stemming, not necessary
    #df['text_'] = df['text_'].apply(lambda x: stem_words(x))
    

    df["text"] = df["text"].apply(lambda text: lemmatize_words(text))
    df["text"] = df["text"].apply(preprocess)
    df.to_csv("processed_Yelp_Food_Reviews_Dataset.csv")
else:
    df = pd.read_csv('processed_Yelp_Food_Reviews_Dataset.csv')

df.dropna(axis=0,inplace=True)
# df.isnull().values.any()
#%% SPLIT DATASET INTO TRAINING & TESTING
review_train, review_test, label_train, label_test = train_test_split(df['text'],df['label'],test_size=0.2)


#%% CONDUCT RANDOMISED SEARCH FOR RANDOM FOREST PIPELINE
params = {'classifier__n_estimators':[100,120], 'classifier__min_samples_split': [2,3], 'classifier__min_samples_leaf': [3,5]}

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfipdf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
])
randomforest_gridsearch = RandomizedSearchCV(pipeline,param_distributions=params, scoring = 'roc_auc', cv=2, verbose = 3) #use randomised search cv instead
randomforest_gridsearch.fit(review_train,label_train)

### PRINT OUT THE BEST SCORES BASED ON TRAINING DATA & CROSS-FOLD EVALAUTION
print(randomforest_gridsearch.best_params_)
print(randomforest_gridsearch.best_score_)

### PRINT OUT THE ACCURACY SCORE ON TEST DATA
print("Test scores are ",randomforest_gridsearch.score(review_test,label_test))


#%% SAVE MODEL PIPELIN
joblib.dump(randomforest_gridsearch, 'RF_fake_review_detector_gs.pkl')

# %%
