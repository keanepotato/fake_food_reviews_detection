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

## Showcase the preprocessing algorithm/ code
### Obtain the preprocessed Dataset

#%%
df = pd.read_csv(r'C:\Users\65932\OneDrive\Desktop\UIT2201\Fake-Reviews-Detection\Preprocessed Fake Reviews Detection Dataset.csv')
df = df.dropna()
df.drop(['category','rating'],axis=1,inplace = True)
df.drop(df.columns[0],axis=1,inplace = True)
indexCG = df[df['label'] == 'CG'].index
indexOR = df[df['label'] == 'OR'].index
indexCG = indexCG[2001:]
indexOR = indexOR[2001:]
df.drop(indexCG, inplace=True)
df.drop(indexOR, inplace=True)
print(len(df))

#review_train, review_test, label_train, label_test = train_test_split(df['text_'],df['label'],test_size=0.35)
#print(review_test)
# print("Old CG " + str(df['label'].value_counts()['CG']))
# print("Old OR " + str(df['label'].value_counts()['OR']))

## Replacing 'CG' and 'OR' values
df['label'] = (df['label'] == 'CG').astype(int)
# print(df.head())
# print("New CG " + str(df['label'].value_counts()[1]))
# print("New OR " + str(df['label'].value_counts()[0]))

df_train, df_test = train_test_split(df, test_size = 0.35)
# print("Train CG " + str(df_train['label'].value_counts()[1]))
# print("Train OR " + str(df_train['label'].value_counts()[0]))

# print("Test CG " + str(df_test['label'].value_counts()[1]))
# print("Test OR " + str(df_test['label'].value_counts()[0]))



ds_dict = {'train' : Dataset.from_pandas(df_train,preserve_index=False),
           'test' : Dataset.from_pandas(df_test,preserve_index=False)}
dataset = DatasetDict(ds_dict)
# dataset['train'] = df_train
# dataset['test'] = df_test
print(dataset)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
 
def tokenize_function(examples):
    return tokenizer(examples["text_"], padding="max_length", truncation=True)
 
tokenized_datasets = dataset.map(tokenize_function, batched=True)
checkpoint = "distilbert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

metric = load_metric("accuracy")
 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=1)
 
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)
 
trainer.train()

model.save_pretrained(r"C:\Users\65932\OneDrive\Desktop\UIT2201\Fake-Reviews-Detection\BERTmodel")
 
# alternatively save the trainer
# trainer.save_model("CustomModels/CustomHamSpam")
 
tokenizer.save_pretrained(r"C:\Users\65932\OneDrive\Desktop\UIT2201\Fake-Reviews-Detection\BERTmodeltoken")