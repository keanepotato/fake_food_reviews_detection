#from sklearn.externals import joblib
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

nltk.download('wordnet')
nltk.download('omw-1.4')

#%% IMPORT THE DATA FROM THE YELP REVIEWS DATASET
df = pd.read_csv(r"C:\Users\65932\OneDrive\Desktop\UIT2201\fake_food_reviews_detection\shortened_Yelp_Food_Reviews_Dataset.csv")
df.dropna(axis=0,inplace=True)

#%% SPLIT THE DATA INTO TRAIN & TEST, AND LOAD DATA INTO THE HUGGINGFACE DATASET
df_train, df_test = train_test_split(df, test_size = 0.2)

ds_dict = {'train' : Dataset.from_pandas(df_train,preserve_index=False),
           'test' : Dataset.from_pandas(df_test,preserve_index=False)}
dataset = DatasetDict(ds_dict)
print(dataset)

#%% IMPORT THE TOKENIZER & HUGGING-FACE BERT MODEL, SET THE TOKENIZER FUNCTION & METRICS
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
checkpoint = "distilbert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
metric = load_metric("accuracy")

#%% TRAIN THE HUGGINGFACE MODEL
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=1)
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)
 
trainer.train()

#%% SAVE THE PRE-TRAINED MODEL & TOKENIZER
model.save_pretrained(r"C:\Users\65932\OneDrive\Desktop\UIT2201\fake_food_reviews_detection\BERTmodel")
 
tokenizer.save_pretrained(r"C:\Users\65932\OneDrive\Desktop\UIT2201\fake_food_reviews_detection\BERTmodeltoken")