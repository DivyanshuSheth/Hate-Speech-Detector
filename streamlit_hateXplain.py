import streamlit as st

import torch   
import torchtext
from torchtext import data    
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
torch.manual_seed(1)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

import random
import os
import base64
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

import spacy
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords

spacy_eng = spacy.load('en')
nltk.download('genesis')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
genesis_ic = wn.ic(genesis, False, 0.0)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


@st.cache
def loadData():
	data = pd.read_csv('train/labeled_data.csv')
	column = 'tweet'
	cleaned_column_ml = 'cleaned_tweet_ml'
	cleaned_column_deep = 'cleaned_tweet_deep'
	data = clean_csv(data, column, cleaned_column_ml, cleaned_column_deep)
	return data


def clean_csv(data, column, cleaned_column_ml, cleaned_column_deep):
  cleaned_sents_ml = []
  cleaned_sents_deep = []
  for sent in data[column]:
    cleaned_ml, cleaned_deep = clean(sent)
    cleaned_sents_ml.append(cleaned_ml)
    cleaned_sents_deep.append(cleaned_deep)
  data[cleaned_column_ml] = cleaned_sents_ml
  data[cleaned_column_deep] = cleaned_sents_deep
  return data


def clean(sent):
  stopw = stopwords.words('english')
  lem = nltk.wordnet.WordNetLemmatizer()
  tokens = sent.split(' ')
  final_tokens_ml = []
  final_tokens_deep = []
  for token in tokens:
    if (not token.startswith('http') and not token.startswith('"@') and not token.startswith('#') and not token.startswith('!') and not token.startswith('&') and not token.startswith('@') and not token.startswith('RT')):
      final_tokens_deep.append(lem.lemmatize(token))
      if token not in stopw:
        final_tokens_ml.append(lem.lemmatize(token)) 
  cleaned_ml =  " ".join(final_tokens_ml).lower() 
  cleaned_ml = re.sub(r'[^\w\s]', '', cleaned_ml)
  cleaned_ml = ''.join([s for s in cleaned_ml if not s.isdigit()])
  cleaned_deep = " ".join(final_tokens_deep).lower()
  cleaned_deep = re.sub(r'[^\w\s]', '', cleaned_deep)
  cleaned_deep = ''.join([s for s in cleaned_deep if not s.isdigit()])
  return cleaned_ml, cleaned_deep

  
def split(data, model_type):
	x_column = 'cleaned_tweet_deep' if model_type == 'BiLSTM' else 'cleaned_tweet_ml'
	X = data[x_column]
	y = data['class']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
	vectorizer = CountVectorizer()
	vectorizer.fit(X_train)
	X_train = vectorizer.transform(X_train)
	X_test = vectorizer.transform(X_test)

	return X_train, X_test, y_train, y_test, vectorizer


@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, y_train, y_test):
	tree = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
	tree.fit(X_train, y_train)
	y_pred = tree.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred, output_dict = True)

	return score, report, tree, y_pred


@st.cache(suppress_st_warning=True)
def neuralNet(X_train, X_test, y_train, y_test):
	scaler = StandardScaler(with_mean = False)  
	scaler.fit(X_train)  
	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test)
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	score1 = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred, output_dict = True)
	
	return score1, report, clf, y_pred


@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train, X_test, y_train, y_test):
	clf = KNeighborsClassifier(n_neighbors=10)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred, output_dict = True)

	return score, report, clf, y_pred

# @st.cache(suppress_st_warning=True)
# def BiLSTM(X_train, X_test, y_train, y_test):
# 	return  score, report, clf, y_pred
# @st.cache(suppress_st_warning=True)

def load_model_and_tokenizer():
  tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
  model = AutoModelForSequenceClassification.from_pretrained(
      "Hate-speech-CNERG/bert-base-uncased-hatexplain",
      num_labels = 3,
      output_attentions = True,
      output_hidden_states = False
  )
  model.to(device)
  return model, tokenizer


def predict_for_user_input(text, model, tokenizer):

  sentences = [text]
  input_ids = []
  attention_masks = []

  for sent in sentences:
      encoded_dict = tokenizer.encode_plus(
                          sent,                      
                          add_special_tokens = True, 
                          max_length = 64,           
                          pad_to_max_length = True,
                          return_attention_mask = True,
                          return_tensors = 'pt',     
                    )
      
      input_ids.append(encoded_dict['input_ids'])
      attention_masks.append(encoded_dict['attention_mask'])

  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  batch_size = 1 
  prediction_data = TensorDataset(input_ids, attention_masks)#, labels)
  prediction_sampler = SequentialSampler(prediction_data)
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
  model.to(device)
  model.eval()
  predictions = []

  for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    attention_vectors = np.mean(outputs[1][11][:,:,0,:].detach().cpu().numpy(), axis=1)
    predictions.append(logits)

  return predictions[0], attention_vectors[0]

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}" download="report.csv">here</a>'
    return href


def plot_confidence(pred_proba, model, col = 'g'):
	model = [model]
	plt.rcParams.update({'font.size': 13})
	fig, ax = plt.subplots(figsize = (10, 0.5))
	ax.barh(model, pred_proba, align='center', color = col)
	ax.set_xlabel('Confidence')
	ax.tick_params(
		axis='y',          
		which='both',      
		left=False,      
		right=False,     
		labelleft=False) 

	ax.tick_params(
		axis='x',        
		which='major',   
		top=False,      
		bottom=True)

	plt.xlim(0, 1)  
	st.pyplot(fig, scale = False)


def print_model_characteristics(model, report, f1macro):
	st.header(model)
	df = pd.DataFrame(report).transpose()
	seq = ['Hate Speech', 'Offensive Speech', 'Normal', 'accuracy', 'macro avg', 'weighted avg']
	df.insert(0, '', seq)
	st.write("Model trained using [data](https://github.com/t-davidson/hate-speech-and-offensive-language) and achieved a macro F-1 score of {:.4f}.".format(f1macro))
	st.markdown("Download the complete report " + get_table_download_link(df) + '.', unsafe_allow_html=True)


def get_print_results(vectorizer, clf, model_type = 'ml'):
	st.text("")
	user_prediction_data_txt = st.text_input("Enter the text:")		
	user_prediction_data_ml, user_prediction_data_deep = clean(user_prediction_data_txt)
	user_prediction_data = vectorizer.transform([user_prediction_data_ml]) if not model_type == 'deep' else vectorizer.transform([user_prediction_data_deep])
	pred = clf.predict_proba(user_prediction_data)
	pred_class = np.argmax(pred)
	pred_proba = pred[0][pred_class]
	pred_color = 'g' if pred_proba > 0.75 else 'r' if pred_proba < 0.5 else 'orange'
	pred_text = 'Hate Speech' if pred_class==0 else ('Offensive Speech' if pred_class==1 else 'Normal')
	if user_prediction_data_txt != "":
		st.subheader(f"The Predicted Class is: {pred_text}")  
		plot_confidence(pred_proba, "Decision Tree", pred_color)
		display_legend("ml")


def print_attentions(user_prediction_data, tokenizer, attention_vectors):

	if user_prediction_data != "":
		st.subheader("Attention visualization:")
	tokens = tokenizer.tokenize(user_prediction_data)
	tokens_colors = []
	sent_length = len(tokens)
	for i, token in enumerate(tokens):
		if attention_vectors[i + 1] * sent_length >= 0.75:
			tokens_colors.append('#FF0000') # Red
		elif attention_vectors[i + 1] * sent_length >= 0.5:
			tokens_colors.append('#FF8700') # Orange
		elif attention_vectors[i + 1] * sent_length >= 0.25:
			tokens_colors.append('#FFEB00') # Yellow
		else:
			tokens_colors.append('#A7FF00') # Green
	text = ""
	for i, token in enumerate(tokens):
		text += ('<b><font size="5" color="' + tokens_colors[i] + '">' + token + ' </font></b>')
	st.markdown(text, unsafe_allow_html=True)

def display_legend(model = 'ml'):
	if model == "BERT":
		st.markdown("<span>Info:<br>Attention visualization on words:  Red (large attention) > Orange > Yellow > Green (less attention)<br>Confidence score:  0.0 to 1.0 - low prediction confidence to high prediction confidence</span>", unsafe_allow_html = True)
	else:
		st.markdown("<span>Info:<br>Confidence score:  0.0 to 1.0 - low prediction confidence to high prediction confidence</span>", unsafe_allow_html = True)



def main():
	st.title("Hate Speech Detector")
	data = loadData()
	X_train, X_test, y_train, y_test, vectorizer = split(data, 'ml')

	choose_model = st.sidebar.selectbox("Choose the Model",
		["BERT", "K-Nearest Neighbours", "Multi-Layer Perceptron", "Decision Tree"])

	if (choose_model == "BERT"):
		st.header("BERT")
		st.write("Model trained using [this data](https://github.com/punyajoy/HateXplain)")
		mod, tok = load_model_and_tokenizer()
		user_prediction_data = st.text_input("Enter the text:")
		pred, attention_vectors = predict_for_user_input(user_prediction_data, mod, tok)
		
		print_attentions(user_prediction_data, tok, attention_vectors)
		probs = softmax(pred)
		pred_class = np.argmax(probs)
		pred_proba = probs[0][pred_class]
		pred_color = 'g' if pred_proba > 0.75 else 'r' if pred_proba < 0.5 else 'orange'
		pred_text = 'Hate Speech' if pred_class==0 else ('Normal' if pred_class==1 else 'Offensive Speech')
		if user_prediction_data != "":
			st.subheader(f"The Predicted Class is: {pred_text}") 
			plot_confidence(pred_proba, "Decision Tree", pred_color)
			display_legend("BERT")

	elif(choose_model == "Decision Tree"):
		score, report, tree, y_pred = decisionTree(X_train, X_test, y_train, y_test)

		f1macro = metrics.f1_score(y_test, y_pred, average = 'macro')
		print_model_characteristics("Decision Tree", report, f1macro)

		get_print_results(vectorizer, tree)

	elif(choose_model == "Multi-Layer Perceptron"):
		score, report, clf, y_pred = neuralNet(X_train, X_test, y_train, y_test)
		
		f1macro = metrics.f1_score(y_test, y_pred, average = 'macro')
		print_model_characteristics("Multi-Layer Perceptron", report, f1macro)

		get_print_results(vectorizer, clf)

	elif(choose_model == "K-Nearest Neighbours"):
		score, report, clf, y_pred = Knn_Classifier(X_train, X_test, y_train, y_test)
		
		f1macro = metrics.f1_score(y_test, y_pred, average = 'macro')
		print_model_characteristics("K-Nearest Neighbours", report, f1macro)

		get_print_results(vectorizer, clf)

	# if st.button('Info'):
	# 	display_legend(choose_model)
	footnote = "<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><span display='block' text-align='center'><bold><font size=\"-2\" color=\"#A9A9A9\">Made by Divyanshu Sheth</font></bold><span>"
	st.sidebar.markdown(footnote, unsafe_allow_html = True)

if __name__ == "__main__":
	main()