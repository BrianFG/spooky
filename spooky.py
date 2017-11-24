import nltk 
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import spacy 
from functools import reduce
from sklearn import tree
import graphviz 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from spacy import attrs
from spacy.symbols import VERB, NOUN, ADV, ADJ
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import cPickle as pickle
import os.path


def intersect(a, b):
    return list(set(a) & set(b))


def most_frequent(dic, limit):
	li = []
	for key in dic:
		li.append(dic[key].index.values.tolist()[:limit])
	return reduce((lambda x, y: intersect(x,y)) , li)

def tokenize(text):
	clean_text = re.sub(r'["]|[,]|[.]|[:]|[?]|[!]|[;]|[(]|[)]', ' ', text.lower())
	return clean_text.split()

def tokenize_filter(text, words):
	clean_text = re.sub(r'["]|[,]|[.]|[:]|[?]|[!]|[;]|[(]|[)]', ' ', text.lower())
	tokens = clean_text.split()
	return list(filter(lambda x: x not in words, tokens))

def text_owner(text, frequencies):
	tokens = tokenize(text)
	pred = pd.DataFrame(columns=['Prediction'])
	for key in frequencies:
		intersected = frequencies.get(key).filter(items=tokens)
		##Avoid NaN values
		if len(intersected) == 0:
			pred.loc[key] = [-0.1]
		else:
			pred.loc[key] = [intersected.sum()]
	if pred.max(axis=0)[0] < 0:
		return "-1"
	else:
		return pred.idxmax(axis=0)[0]

def create_plots(frequencies, limit, name):
	for key in frequencies:
		frequencies.get(key)[:limit].plot(kind="bar")
		plt.savefig(key +  name + ".png")
		plt.clf()

def accuracy(matrix):
	return (matrix * np.eye(len(matrix))).values.sum() / matrix.values.sum()


def plot_confusion_matrix(matrix):
	matrix =  matrix / matrix.sum(axis=1)
	plt.matshow(matrix, cmap=plt.cm.Blues)
	plt.colorbar()
	tick_marks = np.arange(len(matrix.columns))
	plt.xticks(tick_marks, matrix.columns, rotation=45)
	plt.yticks(tick_marks, matrix.index)
	plt.ylabel(matrix.index.name)
	plt.xlabel(matrix.columns.name)
	#plt.colorbar()
	#plt.show()

def token_group(group):
	group_tokens = {}
	for key, value in group:
		all_text = value["text"].str.cat(sep=" ")
		tokens = tokenize(all_text)
		group_tokens[key] = tokens	
	return group_tokens

def main():

	lines = pd.read_csv("./train.csv")
	authors = lines.groupby("author")
	
	author_frequency = {}
	author_tokens = {}
	for key, value in authors:
		all_text = value["text"].str.cat(sep=" ")
		tokens = tokenize(all_text)
		author_tokens[key] = tokens



	for key, value in authors:
		tokens = author_tokens[key]
		frequencies = pd.Series(tokens).value_counts()
		author_frequency[key] =frequencies.div(len(tokens))  

	print ("Saving word frequency distibution...")
	create_plots(author_frequency, 30, "freq")
	print ("Completed.")



	lines_lenght = len(lines)
	train_set = lines[:int(lines_lenght*.75)]
	test_set  = lines[int(lines_lenght*.75)+1:]




	y_actual = test_set["author"]
	deleted_words = []
	accuracies = []

	for i in range (0,201,5):
		frequencies = {}
		rep = most_frequent(author_frequency, i)
		for key, value in authors:
			tokens = list(filter(lambda x: x not in rep, author_tokens[key]))
			counts = pd.Series(tokens).value_counts()
			frequencies[key] = counts.div(len(tokens))
		y_pred = test_set["text"].apply(lambda x: text_owner(x,frequencies))
		confusion_matrix = pd.crosstab(y_actual, y_pred)
		confusion_matrix.index.name = "actual"
		confusion_matrix.columns.name = "predicted"
		acc = accuracy(confusion_matrix)
		print (str(len(rep)) + " ignored words, accuracy: " + str(acc))
		accuracies.append(acc)
		deleted_words.append(rep)
		plot_confusion_matrix(confusion_matrix)
		plt.title("Confusion matrix. " + str(len(rep)) + " most common words ignored" , y=1.18)
		plt.savefig( "CM_" + str(len(rep))  +"ignored.png")
		plt.clf()
		plt.close()
		


	plt.plot(map(lambda x: len(x), deleted_words), accuracies )
	#plt.xticks(map(lambda x: len(x), deleted_words))
	plt.xlabel("#Deleted words")
	plt.ylabel("Accuracy")
	plt.title("Accuracy increments")
	plt.savefig("accuracy.png")


def calculate_frequencies(df, ignored):
	authors = df.groupby("author")
	author_tokens= token_group(authors)
	author_frequency = {}
	for key, value in authors:
		tokens = author_tokens[key]
		frequencies = pd.Series(tokens).value_counts()
		author_frequency[key] =frequencies.div(len(tokens)) 
	rep = most_frequent(author_frequency, ignored)
	frequencies = {}
	for key, value in authors:
		tokens = list(filter(lambda x: x not in rep, author_tokens[key]))
		counts = pd.Series(tokens).value_counts()
		frequencies[key] = counts.div(len(tokens))

	return frequencies

def process_data(df):

	frequencies = {}
	if os.path.exists("./frequencies.p"):
		frequencies = pickle.load( open( "./frequencies.p", "rb" ) )
	else:
		frequencies = calculate_frequencies(df, 235)
		pickle.dump(frequencies, open( "frequencies.p", "wb" ))

	
	

	maping = {"EAP":1 , "HPL": 2, "MWS":3 , "-1":0}

	npl = spacy.load("en")
	df["freq_p"] = df["text"].apply(lambda x: text_owner(x,frequencies))
	df["freq_pred"] = df["freq_p"].apply(lambda x: maping[x])
	df["npl"] = df["text"].apply(lambda x : npl(x.decode("UTF-8")))
	df["tokens"] = df["npl"].str.len()
	df["words"] = df["npl"].apply(lambda x: len([token for token in x if token.is_stop != True and token.is_punct != True ]))
	df["sents"] = df["npl"].apply(lambda x: len([sent for sent in x.sents]))
	df["punctuations"] = df["npl"].apply(lambda x : len([token for token in x if token.is_punct]))
	df["commas"] = df["npl"].apply(lambda x : len([token for token in x if token.string.strip() == ","]))
	df["words_per_sentence"] = df["words"]/df["sents"]
	df["puncts_per_sentence"] = df["punctuations"]/df["sents"]
	df["commas_per_sentence"] = df["commas"]/df["sents"]
	df["pos_counts"] = df["npl"].apply(lambda x: x.count_by(attrs.POS))
	df["nouns"] = df["pos_counts"].apply(lambda x: x[NOUN.numerator] if NOUN in x else 0)
	df["noun_freq"] = df["nouns"]/df["words"]
	df["verbs"] = df["pos_counts"].apply(lambda x: x[VERB.numerator] if VERB in x else 0)
	df["verb_freq"] = df["verbs"]/df["words"]
	df["adjs"] = df["pos_counts"].apply(lambda x: x[ADJ.numerator] if ADJ in x else 0)
	df["adj_freq"] = df["adjs"]/df["words"]
	df["advs"] = df["pos_counts"].apply(lambda x: x[ADV.numerator] if ADV in x else 0)
	df["adv_freq"] = df["advs"]/df["words"]
	return df


def plot_group_data(gp):
	groups = gp.groups.keys()

	for feature in ["noun_freq" , "adj_freq" , "verb_freq" , "adv_freq" ]:
		for key, value in gp:
			vals = 1/value[feature]
			vals = vals.replace([np.inf, -np.inf], np.nan)
			vals = vals.dropna()
			verbs = sorted(vals)
			fit = stats.norm.pdf(verbs, np.mean(verbs), np.std(verbs)) 
			#plt.plot(verbs,fit,'-o')
			plt.plot(verbs,fit) 
		plt.legend(groups)
		#plt.show()  

	for feature in ["words_per_sentence" , "commas_per_sentence" , "puncts_per_sentence" ]:
		for key, value in gp:
			vals = value[feature]
			vals = vals.replace([np.inf, -np.inf], np.nan)
			vals = vals.dropna()
			verbs = sorted(vals)
			fit = stats.norm.pdf(verbs, np.mean(verbs), np.std(verbs)) 
			#plt.plot(verbs,fit,'-o')
			plt.plot(verbs,fit) 
		plt.legend(groups)
		#plt.show()    



def create_tree():

	df = None
	if os.path.exists("./authorsDF.p"):
		df = pd.read_pickle("./authorsDF.p")
	else:
		df = pd.read_csv("./train.csv")
		df = process_data(df)
		df = df.replace([np.inf, -np.inf], np.nan)
		df = df.dropna()
		df = df.drop(["npl"], axis=1)
		df.to_pickle("./authorsDF.p")

	

	


	grouped = df.groupby("author")
	plot_group_data(grouped)




	cols = ["freq_pred" , "words_per_sentence"  ,"puncts_per_sentence" , "commas_per_sentence", "verb_freq" , "adj_freq"]


	train_X, test_X, train_Y, test_Y = train_test_split(df[cols], df["author"], random_state=1)

	
	
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1)
	clf.fit(train_X, train_Y)
	prediction = clf.predict(test_X)
	print accuracy_score(test_Y, prediction)

	model = tree.DecisionTreeClassifier()
	model.fit(train_X, train_Y)
	#print model

	dot_data = tree.export_graphviz(model, out_file=None,  feature_names=cols,  class_names=["EAP" , "HPL" , "MWS"],  
                         filled=True, rounded=True,  
                         special_characters=True) 
	#graph = graphviz.Source(dot_data)
	#graph.render("author") 
	#print graph

	y_predict = model.predict(test_X)
	accuracy  =accuracy_score(test_Y, y_predict)
	print accuracy



def lemma_freq(lemmas, frequencies):
	pred = pd.DataFrame(columns=['Prediction'])
	for key in frequencies:
		intersected = frequencies.get(key).filter(items=lemmas)
		if len(intersected) < 1:
			pred.loc[key] = 0.0
		else:
			pred.loc[key] = [intersected.sum()]
	return pred.idxmax(axis=0)[0]

def spacy_predictions():
	df = pd.read_csv("./train.csv")
	df = df.ix[np.random.choice(df.index, 1000)]
	nlp = spacy.load("en")
	authors = df.groupby("author")
	group_tokens = {}
	for key, value in authors:
		all_text = value["text"].str.cat(sep=" ")
		tokens = nlp(all_text.decode("UTF-8"))
		lemmas = [token for token in tokens if not (token.is_stop or token.is_punct)]
		group_tokens[key] = lemmas

	frequencies = {}
	for key, value in authors:
		tokens = group_tokens[key]
		counts = pd.Series(tokens).value_counts()
		frequencies[key] = counts.div(len(tokens))

	df["nlp"] = df["text"].apply(lambda x : nlp(x.decode("UTF-8")))
	df["lemmas"] = df["nlp"].apply(lambda x: [token for token in x])


	cols = ["lemmas"]
	train_X, test_X, train_Y, test_Y = train_test_split(df[cols], df["author"], random_state=1)

	pred_Y = test_X["lemmas"].apply(lambda x: lemma_freq(x, frequencies))
	print accuracy_score(test_Y, pred_Y)


pd.options.mode.chained_assignment = None
create_tree()

