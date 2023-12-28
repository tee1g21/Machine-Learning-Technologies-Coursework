# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
..
	/////////////////////////////////////////////////////////////////////////
	//
	// (c) Copyright University of Southampton, 2019
	//
	// Copyright in this software belongs to IT Innovation Centre of
	// Gamma House, Enterprise Road, Southampton SO16 7NS, UK.
	//
	// This software may not be used, sold, licensed, transferred, copied
	// or reproduced in whole or in part in any manner or form or in or
	// on any media by any person other than in accordance with the terms
	// of the Licence Agreement supplied with the software, or otherwise
	// without the prior written consent of the copyright owners.
	//
	// This software is distributed WITHOUT ANY WARRANTY, without even the
	// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
	// PURPOSE, except where stated in the Licence Agreement supplied with
	// the software.
	//
	// Created By : Stuart E. Middleton
	// Created Date : 2019/10/31
	// Created for Project: Teaching
	//
	/////////////////////////////////////////////////////////////////////////
	//
	// Dependancies: None
	//
	/////////////////////////////////////////////////////////////////////////
	'''

Data pre-processing and feature selection (support solution for lecture)

Pre-requisites
- dataset >> 20_newsgroups_corpus.json
- py -m pip install nltk
- py -m pip install sklearn
- py -m pip install pandas

"""

import os, sys, logging, traceback, codecs, datetime, copy, time, ast, math, re, random, shutil, json, csv, multiprocessing, subprocess
import nltk, sklearn, pandas
from nltk.util import ngrams
from sklearn.ensemble import RandomForestClassifier

def tokenize_posts( dataset = {} ) :

	for dict_post in dataset :
		# get text of post in its raw form
		str_text = dict_post['body:text']

		# tokenize using newlines into sentence (sent) candidates
		list_sents = str_text.split('\n')

		# tokenize sent candidates using nltk sentence tokenizer, which will look to break up text further using period type patterns
		nIndexSent = 0
		while nIndexSent < len(list_sents) :
			str_sent = list_sents[nIndexSent]
			list_new_sents = nltk.tokenize.sent_tokenize( text = str_sent )
			if len(list_new_sents) > 0 :
				list_sents[nIndexSent] = list_new_sents[0]
				for str_sent in list_new_sents[1:] :
					list_sents.insert( nIndexSent + 1, str_sent )
					nIndexSent = nIndexSent + 1
			else :
				list_sents[nIndexSent] = ''
			nIndexSent = nIndexSent + 1

		# tokenize each sent using nltk word tokenizer
		list_sent_tokens = []
		for str_sent in list_sents :
			list_tokens = nltk.tokenize.word_tokenize( text = str_sent, preserve_line = True )
			list_sent_tokens.append( list_tokens )

		# add token sets to dataset
		dict_post['body:sent_tokens'] = list_sent_tokens

	print( 'Snapshot #2 = \n', dataset[0]['body:sent_tokens'][8] )


def POS_tag_posts( dataset = {} ) :

	for dict_post in dataset :
		# get token sets for each sent
		list_sent_tokens = dict_post['body:sent_tokens']

		# POS token set
		list_sent_pos_sets = nltk.tag.pos_tag_sents( sentences = list_sent_tokens )

		# add token sets to dataset
		dict_post['body:sent_pos'] = list_sent_pos_sets

	print( 'Snapshot #3 = \n', dataset[0]['body:sent_pos'][8] )

def NER_tag_posts( dataset = {} ) :

	for dict_post in dataset :
		# get POS tags for each sent
		list_sent_pos_sets = dict_post['body:sent_pos']

		# NER chunk a POS tagged sent
		list_sent_ner = []
		for list_sent_pos in list_sent_pos_sets :
			dict_NER = {}
			tree_sent = nltk.ne_chunk( list_sent_pos )
			for leaf in tree_sent :
				if isinstance( leaf, nltk.tree.Tree ) :
					list_tokens = []
					list_pos = leaf.leaves()
					for ( str_token, str_pos ) in list_pos :
						list_tokens.append( str_token )
					str_NER_phrase = u' '.join( list_tokens )
					str_NER_type = leaf.label()
					if not str_NER_type in dict_NER :
						dict_NER[ str_NER_type ] = []
					dict_NER[ str_NER_type ].append( str_NER_phrase )
			list_sent_ner.append( dict_NER )

		# add token sets to dataset
		dict_post['body:sent_ner'] = list_sent_ner

	print( 'Snapshot #4 = \n', dataset[0]['body:sent_ner'][8] )

def generate_ngrams( dataset = {}, min_gram = 2, max_gram = 3, allow_pos = True ) :

	for dict_post in dataset :
		# get POS tags for each sent
		list_sent_pos_sets = dict_post['body:sent_pos']

		# generate ngram features for (a) tokens (b) POS sequences
		list_sent_ngrams = []
		for list_sent_pos in list_sent_pos_sets :
			list_tokens = []
			list_pos = []
			for (str_token, str_pos) in list_sent_pos :
				list_tokens.append( str_token )
				list_pos.append( str_pos )

			list_all_ngrams = []
			for nGram in range(min_gram, max_gram+1) :
				list_ngram = list( ngrams( sequence = list_tokens, n = nGram ) )

				# convert token list to a phrase
				for i in range(len(list_ngram)) :
					list_ngram[i] = u' '.join( list_ngram[i] )
				list_all_ngrams.extend( list_ngram )

				if allow_pos == True :
					list_ngram = list( ngrams( sequence = list_pos, n = nGram ) )

					# convert POS list to a phrase
					for i in range(len(list_ngram)) :
						list_ngram[i] = u' '.join( list_ngram[i] )
					list_all_ngrams.extend( list_ngram )

			list_sent_ngrams.append( list_all_ngrams )

		# add token sets to dataset
		dict_post['body:sent_ngrams'] = list_sent_ngrams

	print( 'Snapshot #5 = \n', dataset[0]['body:sent_ngrams'][8] )


def index_features( dataset = {}, list_stopwords = [], allow_pos = True, allow_ngrams = True, allow_NER = True ) :
	dict_feature_index = {}
	list_features = []
	nFeatureID = 0
	for dict_post in dataset :

		list_sent_pos_sets = dict_post['body:sent_pos']
		list_sent_ngrams = dict_post['body:sent_ngrams']
		list_sent_ner = dict_post['body:sent_ner']

		# add unigram tokens (stoplist filtered) and POS
		for list_sent_pos in list_sent_pos_sets :
			for (str_token,str_pos) in list_sent_pos :

				if (not str_token.lower() in list_stopwords) and (not str_token.lower() in dict_feature_index) :
					str_feature = str_token.lower()
					dict_feature_index[str_feature] = nFeatureID
					list_features.append(str_feature)
					nFeatureID = nFeatureID + 1


				if (allow_pos == True) and (not str_pos in dict_feature_index) :
					str_feature = str_pos
					dict_feature_index[str_feature] = nFeatureID
					list_features.append(str_feature)
					nFeatureID = nFeatureID + 1

		# add ngram phrases
		for list_all_ngrams in list_sent_ngrams :
			for str_phrase in list_all_ngrams :
				if (allow_ngrams == True) and (not str_phrase.lower() in list_stopwords) and (not str_phrase.lower() in dict_feature_index) :
					str_feature = str_phrase.lower()
					dict_feature_index[str_feature] = nFeatureID
					list_features.append(str_feature)
					nFeatureID = nFeatureID + 1

		# add NER phrases
		for dict_NER in list_sent_ner :
			for str_NER_type in dict_NER :
				for str_phrase in dict_NER[str_NER_type] :
					if (allow_NER == True) and (not str_phrase.lower() in list_stopwords) and (not str_phrase.lower() in dict_feature_index) :
						str_feature = str_phrase.lower()
						dict_feature_index[str_feature] = nFeatureID
						list_features.append(str_feature)
						nFeatureID = nFeatureID + 1

	print( 'Snapshot #6 =' )
	nCount = 0
	for str_feature in dict_feature_index :
		print( '\t', str_feature, ' = ', dict_feature_index[str_feature] )
		nCount = nCount + 1
		if nCount > 10 :
			break

	return ( dict_feature_index, list_features )


def calc_count_vector( dataset = {}, dict_index = {} ) :

	# compile an index of group names (these will be our 'documents')
	index_group = {}
	list_groups = []
	nGroupIndex = 0
	for dict_post in dataset :
		if not dict_post['post:group'] in index_group :
			index_group[ dict_post['post:group'] ] = nGroupIndex
			list_groups.append( dict_post['post:group'] )
			nGroupIndex = nGroupIndex + 1

	# create count vector (rows = document, columns = features, cells = frequency of occurance) with 0 freq
	list_count_vector = []
	for nGroupIndex in range(len(index_group)) :
		list_count_vector.append( [0] * len(dict_index) )

	# add freq to each occurance of a feature in a document
	for dict_post in dataset :

		# get post group (document type)
		str_group = dict_post['post:group']
		nGroupIndex = index_group[str_group]

		# get features in post
		list_sent_pos_sets = dict_post['body:sent_pos']
		list_sent_ner = dict_post['body:sent_ner']
		list_sent_ngrams = dict_post['body:sent_ngrams']

		# add unigram tokens and POS
		for list_sent_pos in list_sent_pos_sets :
			for (str_token,str_pos) in list_sent_pos :

				str_feature = str_token.lower()
				if str_feature in dict_index :
					list_count_vector[ nGroupIndex ][ dict_index[str_feature] ] += 1

				str_feature = str_pos
				if str_feature in dict_index :
					list_count_vector[ nGroupIndex ][ dict_index[str_feature] ] += 1

		# add ngram phrases
		for list_all_ngrams in list_sent_ngrams :
			for str_phrase in list_all_ngrams :
				str_feature = str_phrase.lower()
				if str_feature in dict_index :
					list_count_vector[ nGroupIndex ][ dict_index[str_feature] ] += 1

		# add NER phrases
		for dict_NER in list_sent_ner :
			for str_NER_type in dict_NER :
				for str_phrase in dict_NER[str_NER_type] :
					str_feature = str_phrase.lower()
					if str_feature in dict_index :
						list_count_vector[ nGroupIndex ][ dict_index[str_feature] ] += 1

	return ( list_count_vector, list_groups )

def calc_test_train_matrix( dataset = {}, list_group = [], set_features = set([]) ) :

	# compile an index of group names
	index_group = {}
	nGroupIndex = 0
	for str_group in list_group :
		index_group[str_group] = nGroupIndex
		nGroupIndex = nGroupIndex + 1

	# compile an index of features
	index_features = {}
	nFeatureIndex = 0
	for str_feature in set_features :
		index_features[str_feature] = nFeatureIndex
		nFeatureIndex = nFeatureIndex + 1

	# create feature and label vectors
	X = []
	Y = []
	list_feature_set = list( set_features )
	for dict_post in dataset :

		# get post group (document type)
		str_group = dict_post['post:group']
		nGroupIndex = index_group[str_group]
		Y.append( nGroupIndex )

		# get features in post
		list_sent_pos_sets = dict_post['body:sent_pos']
		list_sent_ner = dict_post['body:sent_ner']
		list_sent_ngrams = dict_post['body:sent_ngrams']
		list_freq_vector = [0] * len(list_feature_set)

		# add unigram tokens and POS
		for list_sent_pos in list_sent_pos_sets :
			for (str_token,str_pos) in list_sent_pos :

				str_feature = str_token.lower()
				if str_feature in index_features :
					list_freq_vector[ index_features[str_feature] ] += 1

				str_feature = str_pos
				if str_feature in index_features :
					list_freq_vector[ index_features[str_feature] ] += 1

		# add ngram phrases
		for list_all_ngrams in list_sent_ngrams :
			for str_phrase in list_all_ngrams :
				str_feature = str_phrase.lower()
				if str_feature in index_features :
					list_freq_vector[ index_features[str_feature] ] += 1

		# add NER phrases
		for dict_NER in list_sent_ner :
			for str_NER_type in dict_NER :
				for str_phrase in dict_NER[str_NER_type] :
					str_feature = str_phrase.lower()
					if str_feature in index_features :
						list_freq_vector[ index_features[str_feature] ] += 1

		X.append( list_freq_vector )

	return ( X,Y )

def load_raw_dataset( filename = None ) :

	read_handle = codecs.open( filename, 'r', 'utf-8', errors = 'replace' )
	list_lines = read_handle.readlines()
	read_handle.close()

	dataset_raw = []
	for str_line in list_lines :
		dict_post = json.loads( str_line )
		dataset_raw.append( dict_post )

	# debug limit size of text to process. randomize to avoid all being same topic. keep 1st post for lecture examples.
	'''
	first_entry = dataset_raw[0]
	dataset_raw = dataset_raw[1:]
	random.shuffle( dataset_raw )
	dataset_raw = dataset_raw[:1000]
	dataset_raw.insert( 0, first_entry )
	'''
	#end debug

	print( 'Snapshot #1 = \n', dataset_raw[0]['body:text'][:600] )

	return dataset_raw

################################
# main
################################

# only execute if this is the main file
if __name__ == '__main__' :

	# make logger (global to STDOUT)
	LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
	logger = logging.getLogger( __name__ )
	logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
	logger.info('logging started')

	#
	# load raw dataset
	#

	strFile = '20_newsgroups_corpus.json'
	logger.info( 'loading dataset: ' + strFile )
	listDatasetRaw = load_raw_dataset( filename = strFile )
	logger.info( 'Number of posts in raw dataset = ' + repr(len(listDatasetRaw)) )

	# try different settings yourself
	allow_pos = False
	allow_NER = True
	allow_ngrams = True

	#
	# tokenize text (sentence and word tokenization)
	# find out more: https://www.nltk.org/_modules/nltk/tokenize.html
	# find out more: https://stanfordnlp.github.io/CoreNLP/
	#

	tokenize_posts( dataset = listDatasetRaw )

	#
	# POS tag text
	# find out more: https://www.nltk.org/book/ch05.html
	# find out more: https://stanfordnlp.github.io/CoreNLP/
	#

	POS_tag_posts( dataset = listDatasetRaw )

	#
	# NER tag text
	# find out more: https://www.nltk.org/book/ch07.html
	# find out more: https://www.nltk.org/_modules/nltk/tree.html
	# find out more: https://stanfordnlp.github.io/CoreNLP/
	#

	NER_tag_posts( dataset = listDatasetRaw )

	#
	# create n-gram features (bigrams and trigrams)
	# find out more: https://www.nltk.org/_modules/nltk/util.html
	#

	generate_ngrams( dataset = listDatasetRaw, min_gram = 2, max_gram = 3, allow_pos = allow_pos )

	#
	# index all available categorical features (tokens, POS, ngrams, NER) so we can work with a count vector index, not text
	# provide a domain specific stoplist to remove tokens with little discriminating value (common words, punctuation, symbols)
	# find out more: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
	#

	listStopTokens = nltk.corpus.stopwords.words()
	listStopTokens.extend( [ ':', ';', '[', ']', '"', "'", '(', ')', '.', '?', '#', '@', ',', '`', '``', "''", "'", '-', '--', '*', '|', '>', '<', '=', '%', '$', '+', '/', '\\' ] )

	( dict_feature_index, list_features ) = index_features( dataset = listDatasetRaw, list_stopwords = listStopTokens, allow_pos = allow_pos, allow_NER = allow_NER, allow_ngrams = allow_ngrams )

	print( '\nnumber of features = ', len(dict_feature_index) )

	#
	# create a count vector. row = document. column = feature.
	# find out more: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
	#

	( array_count_vector, list_group ) = calc_count_vector( dataset = listDatasetRaw, dict_index = dict_feature_index )

	print( '\nnumber of groups = ', len(array_count_vector) )

	transformer = sklearn.feature_extraction.text.TfidfTransformer( smooth_idf = False, use_idf = True )
	transformer.fit( array_count_vector )

	df_idf = pandas.DataFrame( data = transformer.idf_, index = list_features, columns = ['idf_weights'] )
	df_idf = df_idf.sort_values( by=['idf_weights'], ascending = False )

	print( 'Snapshot #7 IDF (top 20) for corpus = ', df_idf[0:20] )

	tf_idf = transformer.transform( array_count_vector )

	print( 'Snapshot #8.1 TF-IDF (top 20) for group ', list_group[0] )

	tf_idf_vector = tf_idf[0]
	df_tf_idf = pandas.DataFrame( tf_idf_vector.T.todense(), index=list_features, columns=['tfidf'] )
	df_tf_idf = df_tf_idf.sort_values( by=["tfidf"], ascending=False )
	print( df_tf_idf[:20], '\n' )

	print( 'Snapshot #8.2 TF-IDF (top 20) for group ', list_group[1] )

	tf_idf_vector = tf_idf[1]
	df_tf_idf = pandas.DataFrame( tf_idf_vector.T.todense(), index=list_features, columns=['tfidf'] )
	df_tf_idf = df_tf_idf.sort_values( by=["tfidf"], ascending=False )
	print( df_tf_idf[:20], '\n' )

	print( 'Snapshot #8.3 TF-IDF (top 20) for group ', list_group[2] )

	tf_idf_vector = tf_idf[2]
	df_tf_idf = pandas.DataFrame( tf_idf_vector.T.todense(), index=list_features, columns=['tfidf'] )
	df_tf_idf = df_tf_idf.sort_values( by=["tfidf"], ascending=False )
	print( df_tf_idf[:20], '\n' )

	#
	# Run experiment with several topN feature selection thresholds
	#

	for nTopN in [ 20, 100, 1000, 10000 ] :

		print( 'Processing topN ', nTopN )

		#
		# prepare a categorical feature vector training set for use with a post topic classifier
		# for each post create a vector with its feature freq count
		# use a feature selection strategy of aggregating the topN TF-IDF features from each document class
		#

		set_features = set([])
		for nGroupIndex in range(len(list_group)) :
			tf_idf_vector = tf_idf[nGroupIndex]
			df_tf_idf = pandas.DataFrame( tf_idf_vector.T.todense(), index=list_features, columns=['tfidf'] )
			df_tf_idf = df_tf_idf.sort_values( by=["tfidf"], ascending=False )
			df_tf_idf = df_tf_idf[:nTopN]
			for str_feature in df_tf_idf.index :
				set_features.add( str_feature )

		#
		# Train random forest classifier (90% training set, 10% test set)
		# find out more: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
		#

		random.shuffle( listDatasetRaw )
		dataset_train = listDatasetRaw[ : int(len(listDatasetRaw)*0.8) ]
		dataset_test = listDatasetRaw[ int(len(listDatasetRaw)*0.8) : ]
		print('Creating training matrix')
		(X_train, Y_train) = calc_test_train_matrix( dataset = dataset_train, list_group = list_group, set_features = set_features )
		print('Creating test matrix')
		(X_test, Y_test) = calc_test_train_matrix( dataset = dataset_test, list_group = list_group, set_features = set_features )

		rf = RandomForestClassifier( n_estimators = 500, max_leaf_nodes = 16, n_jobs = 6 )
		print('RF constructed')
		rf.fit( X_train, Y_train )
		print('RF trained')
		Y_predict = rf.predict( X_test )
		print('RF predicted')

		print( 'Snapshot #9 first 20 posts [top ', nTopN, ' features per document class]' )
		for post_index in range(20) :
			print( '\tPredict ', list_group[ Y_predict[post_index] ], '   [Gold ', list_group[ Y_test[post_index] ], ']' )

		#
		# Compute post classification precision
		#

		nTP = 0
		nFP = 0
		for post_index in range(len(X_test)) :
			if Y_test[post_index] == Y_predict[post_index] :
				nTP += 1
			else :
				nFP += 1
		if nTP + nFP > 0 :
			nP = (1.0 * nTP) / (nTP + nFP)
		else :
			nP = 0.0

		print( '\tTP = ', nTP, ' FP = ', nFP, ' P = ', nP, '\n' )

