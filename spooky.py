import nltk 
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import spacy 
from functools import reduce
from sklearn import tree
from sklearn import svm
import graphviz 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from spacy import attrs
from spacy.symbols import VERB, NOUN, ADV, ADJ , PROPN
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import cPickle as pickle
import os.path
import random
import itertools



## Returns the intersection between 2 lists
def intersect(a, b):
    return list(set(a) & set(b))


## Returns the most frequent values between lists in a dictionary. Takes *limit* from each list
def most_frequent(dic, limit):
	li = []
	for key in dic:
		li.append(dic[key].index.values.tolist()[:limit])
	return reduce((lambda x, y: intersect(x,y)) , li)

## Splits a string into a word list
def tokenize(text):
	clean_text = re.sub(r'["]|[,]|[.]|[:]|[?]|[!]|[;]|[(]|[)]', ' ', text.lower())
	return clean_text.split()


## Slits a string into words list and eliminates words contained in [words] list
def tokenize_filter(text, words):
	clean_text = re.sub(r'["]|[,]|[.]|[:]|[?]|[!]|[;]|[(]|[)]', ' ', text.lower())
	tokens = clean_text.split()
	return list(filter(lambda x: x not in words, tokens))

## Calculates word frequency to predict the test author
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

## Generates word frequency plots
def create_plots(frequencies, limit, name):
	for key in frequencies:
		frequencies.get(key)[:limit].plot(kind="bar")
		plt.savefig(key +  name + ".png")
		plt.clf()

## Calculates acuray from a confusion matrix (crosstab)
def accuracy(matrix):
	return (matrix * np.eye(len(matrix))).values.sum() / matrix.values.sum()


## Plots a confusion matrix
def plot_confusion_matrix(matrix):
	matrix =  matrix / matrix.sum(axis=1)
	plt.matshow(matrix, cmap=plt.cm.BuPu)
	plt.colorbar()
	tick_marks = np.arange(len(matrix.columns))
	plt.xticks(tick_marks, matrix.columns, rotation=45)
	plt.yticks(tick_marks, matrix.index)
	plt.ylabel(matrix.index.name)
	plt.xlabel(matrix.columns.name)
	#plt.colorbar()
	#plt.show()

## Tokenize text by author and returns a dictionary {author : tokens list}
def token_group(group):
	group_tokens = {}
	for key, value in group:
		all_text = value["text"].str.cat(sep=" ")
		tokens = tokenize(all_text)
		group_tokens[key] = tokens	
	return group_tokens


# Some tests to identify the best way to get word frequencies
# Instead of ignoring stop words, it ignores the most frequen words between authors.
def word_frequency_test():
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
		# value (235) based on a previous experiment
		# This will take the 235 most frequent words from each author and ignore intersactions 
		frequencies = calculate_frequencies(df, 235) 
		pickle.dump(frequencies, open( "frequencies.p", "wb" ))

	
	maping = {"EAP":1 , "HPL": 2, "MWS":3 , "-1":0}

	nlp = spacy.load("en")
	df["freq_p"] = df["text"].apply(lambda x: text_owner(x,frequencies))
	df["freq_pred"] = df["freq_p"].apply(lambda x: maping[x])

	df["nlp"]    = df["text"].apply(lambda x : nlp(x.decode("UTF-8")))

	df["tokens"] = df["nlp"].str.len()
	df["sents"]  = df["nlp"].apply(lambda x: len([sent for sent in x.sents]))
	df["words"]  = df["nlp"].apply(lambda x: len([token for token in x if token.is_stop != True and token.is_punct != True ]))
	df["stops"]  = df["nlp"].apply(lambda x: len([token for token in x if token.is_stop ]))
	df["puncts"] = df["nlp"].apply(lambda x: len([token for token in x if token.is_punct]))
	df["commas"] = df["nlp"].apply(lambda x: len([token for token in x if token.string.strip() == ","]))

	#Calculating different features per sentence
	df["Words per sentence"]  = df["words"]/df["sents"]
	df["Puncts per sentence"] = df["puncts"]/df["sents"]
	df["Commas per sentence"] = df["commas"]/df["sents"]
	df["Stops per sentence"]  = df["stops"]/df["sents"]


	#Part of Speech analysis (counts)
	df["pos"]    = df["nlp"].apply(lambda x: x.count_by(attrs.POS))
	df["nouns"]  = df["pos"].apply(lambda x: x[NOUN.numerator] if NOUN in x else 0)
	df["verbs"]  = df["pos"].apply(lambda x: x[VERB.numerator] if VERB in x else 0)
	df["adjs"]   = df["pos"].apply(lambda x: x[ADJ.numerator] if ADJ in x else 0)
	df["advs"]   = df["pos"].apply(lambda x: x[ADV.numerator] if ADV in x else 0)
	df["propns"] = df["pos"].apply(lambda x: x[PROPN.numerator] if PROPN in x else 0)
	
	# part of speech frequency , POS:word ratio
	df["Noun Frequency"]   = df["nouns"]/df["words"]
	df["Verb Frequency"]   = df["verbs"]/df["words"]
	df["Adj Frequency"]    = df["adjs"] /df["words"]
	df["Adverb Frequency"] = df["advs"] /df["words"]
	df["PropN Frequency"]  = df["propns"] /df["words"]


	return df


def plot_group_data(gp):
	groups = gp.groups.keys()

	for feature in ["Noun Frequency" , "Adj Frequency" , "Verb Frequency" , "Adverb Frequency" , "PropN Frequency" ]:
		for key, value in gp:
			values = value[value[feature] >0]
			vals = 1/values[feature]
			verbs = sorted(vals)
			fit = stats.norm.pdf(verbs, np.mean(verbs), np.std(verbs)) 
			#plt.plot(verbs,fit,'ro')
			plt.plot(verbs,fit) 
		plt.legend(groups)
		plt.title(feature)
		plt.xlabel("Word : " + feature.split()[0] + " ratio" )
		plt.ylabel("Frenquency")
		plt.savefig("dist_" + feature + ".png")
		plt.clf() 

	for feature in ["words" , "sents" , "puncts",  "Words per sentence" , "Commas per sentence" , "Puncts per sentence" , "Stops per sentence" ]:
		for key, value in gp:
			vals = value[feature]
			verbs = sorted(vals)
			fit = stats.norm.pdf(verbs, np.mean(verbs), np.std(verbs)) 
			#plt.plot(verbs,fit,'-o')
			plt.plot(verbs,fit) 
		plt.title(feature.replace("_" , " "))
		plt.legend(groups)
		plt.title(feature)
		plt.xlabel(feature )
		plt.ylabel("Frenquency")
		plt.savefig("dist_" +  feature + ".png") 
		plt.clf()



def test_learning():

	df = None
	if os.path.exists("./authorsDF.p"):
		df = pd.read_pickle("./authorsDF.p")
	else:
		df = pd.read_csv("./train.csv")
		#df = df.ix[random.sample(df.index, 100)]
		df = process_data(df)
		df = df.replace([np.inf, -np.inf], np.nan)
		df = df.dropna()
		df = df.drop(["nlp"], axis=1)
		df.to_pickle("./authorsDF.p")

	
	## Uncomment next 2 lines to plot distribution graphs 
	#grouped = df.groupby("author")
	#plot_group_data(grouped)

	## Selected columns for prediction 
	all_cols = ["freq_pred" ,"Noun Frequency" , "Adj Frequency" , "Verb Frequency" , "Adverb Frequency" , "PropN Frequency" ,  "Words per sentence" , "Commas per sentence" , "Puncts per sentence" , "Stops per sentence" ]
	

	#Selecting small random set to print Classification tree
	df_small = df.ix[random.sample(df.index, 1000)]

	# Spliting train and test set 
	train_X, test_X, train_Y, test_Y = train_test_split(df_small[all_cols], df_small["author"], random_state=1)

	# Creating classification tree
	clf_tree = tree.DecisionTreeClassifier()
	clf_tree.fit(train_X, train_Y)

	#Ploting and saving file
	dot_data = tree.export_graphviz(clf_tree, out_file=None,  feature_names=all_cols,  class_names=["EAP" , "HPL" , "MWS"],  
                         filled=True, rounded=True,  
                         special_characters=True) 
	graph = graphviz.Source(dot_data)
	graph.render("ClassTree_autors") 
	y_predict = clf_tree.predict(test_X)
	accuracy  =accuracy_score(test_Y, y_predict)
	print ("Classification tree accuracy: " + str(accuracy))



	print ("Training all data Set")
	##### training all set #####
	#####                  #####


	# Based on graphs and tree plot, the  following features were selected to get best accuracy avoiding overfitting 
	cols = ["freq_pred" , "PropN Frequency" , "Commas per sentence" , "Words per sentence" , "Verb Frequency" , "Noun Frequency" ]
	# Spliting train and test sets
	train_X, test_X, train_Y, test_Y = train_test_split(df[cols], df["author"], random_state=1)

	
	# Creating Decision tree
	clf_tree = tree.DecisionTreeClassifier()
	clf_tree.fit(train_X, train_Y)
	predict_Y = clf_tree.predict(test_X)
	accuracy  = accuracy_score(test_Y, predict_Y)
	print ("Decision tree accuracy: " + str(accuracy))
	confusion_matrix = pd.crosstab(test_Y, predict_Y)
	confusion_matrix.index.name = "Actual"
	confusion_matrix.columns.name = "Predicted"
	plot_confusion_matrix(confusion_matrix)
	plt.title("Decision Tree prediction", y=1.18)
	plt.savefig("CM_DecisionTree.png")
	plt.clf()
	plt.close()

	# Creating multilayer perceptron 
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 10), random_state=1)
	clf.fit(train_X, train_Y)
	predict_Y = clf.predict(test_X)
	accuracy = accuracy_score(test_Y, predict_Y)
	print ("MLP accuracy: " + str(accuracy))
	confusion_matrix = pd.crosstab(test_Y, predict_Y)
	confusion_matrix.index.name = "Actual"
	confusion_matrix.columns.name = "Predicted"
	plot_confusion_matrix(confusion_matrix)
	plt.title("Multilayer Perceptron  prediction", y=1.18)
	plt.savefig("CM_MLP.png")
	plt.clf()
	plt.close()


	# Creating  Support Vector Machine
	clf = svm.SVC(decision_function_shape='ovo')
	clf.fit(train_X, train_Y)
	predict_Y = clf.predict(test_X)
	accuracy = accuracy_score(test_Y, predict_Y)
	print ("SVM accuracy: " + str(accuracy))
	confusion_matrix = pd.crosstab(test_Y, predict_Y)
	confusion_matrix.index.name = "Actual"
	confusion_matrix.columns.name = "Predicted"
	plot_confusion_matrix(confusion_matrix)
	plt.title("Support Vector Machineprediction", y=1.18)
	plt.savefig("CM_SVM.png")
	plt.clf()
	plt.close()



pd.options.mode.chained_assignment = None


test_learning()

