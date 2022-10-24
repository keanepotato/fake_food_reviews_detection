from cProfile import label
import joblib
import gradio as gr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
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
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TextClassificationPipeline
import torch
import numpy as np

### IMPORT BERT MODEL
print("Running prototype")
bert_model = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\65932\OneDrive\Desktop\UIT2201\fake_food_reviews_detection\BERTmodel")
bert_tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\65932\OneDrive\Desktop\UIT2201\fake_food_reviews_detection\BERTmodeltoken")

### CREATING BERT PIPELINE
bert_pipeline = TextClassificationPipeline(model=bert_model, tokenizer=bert_tokenizer,return_all_scores=True)

### DEFINE ML FUNCTION
def classify_review(x):
    score_labels = bert_pipeline(x)[0]
    score_labels.sort(key = lambda x : x['score'], reverse=True)
    print(score_labels)
    label = score_labels[0]['label']
    if label == 'LABEL_1':
        return "Real"
    else:
        return "Fake"


### INITIALISE GRADIO INTERFACE
initialise_gradio = True

if initialise_gradio:
    

    txt_1 = gr.Textbox(label= "Review to be examined")

    ### TEST EXAMPLES
    set_one_second_eg = "Restaurant keeps cold drinks cold, hot drinks hot, cold drinks hot."
    
    set_one_first_eg = "Can't miss stop for the best Fish Sandwich in Pittsburgh."


    iface = gr.Interface(fn = classify_review, 
                        inputs = txt_1, 
                        outputs = 'label', 
                        title = 'Fake Review Detection with Custom Bert Model',
                        description = 'Input any text that you believe to be fake, this will return an output',
                        article = 
                            '''<div>
                                <p> Hit submit after putting your fake review text to see the results.</p>
                            </div>''',
                        examples = [[set_one_first_eg], [set_one_second_eg]],
                        share = True)

    iface.launch(share = True)