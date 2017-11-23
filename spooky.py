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
			pred.loc[key] = [0.00000000001]
		else:
			pred.loc[key] = [intersected.sum()]
		
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
	


def create_tree():
	alls = pd.read_csv("./train.csv")
	df = alls[:320]

	npl = spacy.load("en")
	df["npl"] = df["text"].apply(lambda x : npl(x.decode("UTF-8")))
	df["tokens"] = df["npl"].str.len()
	df["words"] = df["npl"].apply(lambda x: len([token for token in x if token.is_stop != True and token.is_punct != True ]))
	df["sents"] = df["npl"].apply(lambda x: len([sent for sent in x.sents]))
	df["words_per_sentence"] = df["words"]/df["sents"]


	train_X = df[["tokens" , "words" , "words_per_sentence" ]][:250]
	train_Y = df["author"][:250]

	test_X = df[["tokens" , "words" , "words_per_sentence" ]][251:]
	test_Y = df["author"][251:]


	model = tree.DecisionTreeClassifier()
	model.fit(train_X, train_Y)
	#print model

	dot_data = tree.export_graphviz(model, out_file=None) 
	graph = graphviz.Source(dot_data) 
	graph.render("author", view=True) 
	#print graph

	y_predict = model.predict(test_X)
	accuracy  =accuracy_score(test_Y, y_predict)
	print accuracy





create_tree()