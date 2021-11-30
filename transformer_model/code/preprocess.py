import tensorflow as tf
import numpy as np
from functools import reduce
import pandas as pd

def build_vocab(train_descriptions, test_descriptions):
	"""
	Builds vocab from list of sentences
	"""
	tokens = set() # set of all tokens, without repetition

	better_list_of_train_descriptions, better_list_of_test_descriptions = [], []
	for i in [0,1]:
		for sentence in [train_descriptions, test_descriptions][i]:

			# make lower case, remove extra characters
			sentence = sentence.lower()
			sentence = sentence.replace(";", "").replace(":", "").replace("(", "").replace(")", "").replace(".", "")
			sentence = sentence.replace(",", "").replace("/", " ").replace("-", " ").replace("•\t", "").replace("@", "")
			sentence = sentence.replace('"', "").replace("\n", " ").replace("#", "")
			sentence = sentence.replace("  ", " ")

			sentence = sentence.split(" ") # get list of words in sentence
			if i == 0:
				better_list_of_train_descriptions.append(sentence)
			elif i == 1:
				better_list_of_test_descriptions.append(sentence)

			for word in sentence:
				tokens.add(word)

	word_to_id = {word:idx for idx,word in enumerate(tokens, start=1)} # changed from Nate's preprocess to start from 1 so pad token can be 0

	return tokens, word_to_id, better_list_of_train_descriptions, better_list_of_test_descriptions

def convert_to_id(word_to_id, sentences):
	"""
	Convert sentences to indexed 
	"""
	# return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])
	better_sentences = []
	for sentence in sentences:
		better_sentence = [word_to_id[word] for word in sentence]
		better_sentences.append(better_sentence)
	return better_sentences

def get_data(x_train, y_train, x_test, y_test, description):
	"""
	Reads in, parses, and tokenizes the csv's that contain the training and testing data

	:param x_train:     path to csv containing descriptions of clinical interactions
	:param y_train:     path to csv containing diagnoses for those clinical interactions
	:param x_test:      path to csv containing descriptions of clinical interactions
	:param y_test:      path to csv containing diagnoses for those clinical interactions
	:param y_test:      one of AEHEVNT, AEHCOMM, CONCAT
	:return:            (train_descriptions, train_diagnoses, test_descriptions, test_diagnoses)
	"""

	# STEP 1: Read in the description of adverse patient events
	train_descriptions = pd.read_csv(x_train, usecols=[description])
	train_descriptions = [train_descriptions.loc[i,description] for i in range(len(train_descriptions))]
	test_descriptions = pd.read_csv(x_test, usecols=[description])
	test_descriptions = [test_descriptions.loc[i,description] for i in range(len(test_descriptions))]

	# STEP 2: Construct vocabulary, tokenize
	tokens, word_to_id, train_sentences, test_sentences = build_vocab(train_descriptions, test_descriptions)

	# STEP 3: Convert sentences to lists of ID's
	train_descriptions = convert_to_id(word_to_id, train_sentences)
	test_descriptions = convert_to_id(word_to_id, test_sentences)

	# STEP 4: Read in the ultimate patient diagnoses
	train_diagnoses = pd.read_csv(y_train, usecols=["DIAGNOSIS"])
	test_diagnoses = pd.read_csv(y_test, usecols=["DIAGNOSIS"])

	# STEP 5: Convert patient diagnoses to list of integers
	train_diagnoses = [train_diagnoses.loc[i,"DIAGNOSIS"]-1 for i in range(len(train_diagnoses))]
	test_diagnoses = [test_diagnoses.loc[i,"DIAGNOSIS"]-1 for i in range(len(test_diagnoses))]

	return train_descriptions, train_diagnoses, test_descriptions, test_diagnoses, word_to_id
