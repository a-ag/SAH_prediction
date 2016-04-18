from sklearn import svm
import csv
import copy
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import string
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import BernoulliNB

class assignment2_Part1:
	def __init__(self):
		temp = open('Assignment_II/data/pos_examples_happy.txt','r')
		print temp
		self.data_happy = []
		for item in temp:
			result = re.sub(r"http\S+", "", item.encode('ascii','ignore'))
			if 'R' == result[0] and 'T'==result[1]:
				continue
			self.data_happy.append(result)
		# for item in self.data_happy:
		# 	print item

		temp = open('Assignment_II/data/neg_examples_sad.txt','r')
		print temp
		self.data_sad = []
		for item in temp:
			result = re.sub(r"http\S+", "", item.encode('ascii','ignore'))
			if 'R' == result[0] and 'T'==result[1]:
				continue
			self.data_sad.append(result)

		print len(self.data_happy)
		print len(self.data_sad)
		# raw_input("dilns")
	
	def tokenizeDocument(self,stopWordInstruction=True):
		tweetTokenizerInitialization = TweetTokenizer(strip_handles = True, reduce_len=True)
		regExpTokenizer = RegexpTokenizer(r'\w+')
		happyTweetsTokenized = []
		print type(self.data_happy[0])
		stopwords_mine = ([word.encode('ascii','ignore') for word in stopwords.words('english')])
		print stopwords_mine
		print self.data_happy[0]
		for item in self.data_happy:
			temp_tokenized = tweetTokenizerInitialization.tokenize(item)
			temp_tokenized = ([item1.encode('ascii','ignore') for item1 in temp_tokenized if (item1.encode('ascii','ignore')!= 'RT' and item1.encode('ascii','ignore')!='http')] )
			temp_tokenized = " ".join(temp_tokenized)
			temp_tokenized = regExpTokenizer.tokenize(temp_tokenized)

			happyTweetsTokenized.append([item1.encode('ascii','ignore') for item1 in temp_tokenized if item1.encode('ascii','ignore')!= 'RT'])
		#raw_input("Press Enter")
		# for item in happyTweetsTokenized:
		# 	print item
		self.new_happyTweets_tokenized = []
		if stopWordInstruction==True:
			for item in happyTweetsTokenized:
				temp = []
				temp += (word.lower() for word in item if word.lower() not in stopwords_mine)
				self.new_happyTweets_tokenized.append(temp)
		else:
			self.new_happyTweets_tokenized=copy.deepcopy(happyTweetsTokenized)

		# for item in self.new_happyTweets_tokenized:
		# 	print item



		##########################
		#SAD PART NOW
		##########################
		sadTweetsTokenized = []
		for item in self.data_sad:
			temp_tokenized = tweetTokenizerInitialization.tokenize(item)
			temp_tokenized = ([item1.encode('ascii','ignore') for item1 in temp_tokenized if (item1.encode('ascii','ignore')!= 'RT' and item1.encode('ascii','ignore')!='http')] )
			temp_tokenized = " ".join(temp_tokenized)
			temp_tokenized = regExpTokenizer.tokenize(temp_tokenized)

			sadTweetsTokenized.append([item1.encode('ascii','ignore') for item1 in temp_tokenized if item1.encode('ascii','ignore')!= 'RT'])
		#raw_input("Press Enter")
		# for item in happyTweetsTokenized:
		# 	print item
		self.new_sadTweets_tokenized = []
		if stopWordInstruction==True:
			for item in sadTweetsTokenized:
				temp = []
				temp += (word.lower() for word in item if word.lower() not in stopwords_mine)
				self.new_sadTweets_tokenized.append(temp)
		else:
			self.new_sadTweets_tokenized=copy.deepcopy(happyTweetsTokenized)


		# for item in self.new_sadTweets_tokenized:
		# 	print item

	def classifierDataAffect(self):
		#for affect files
		df_sad_tweets = pd.read_csv('tweet_sentiment_sad_RT_removed.csv',sep='\t')
		df_sad_tweets['output'] = 0
		df_new_sad_tweets = df_sad_tweets.ix[:,0:6]
		print df_new_sad_tweets.ix[1:10]
		print df_new_sad_tweets.columns.values
		df_new_sad_tweets['output']=0
		print df_new_sad_tweets.columns.values
		print df_new_sad_tweets.ix[1:10]
		# print df_sad_tweets.ix[1:10,0:7]

		df_happy_tweets = pd.read_csv('tweet_sentiment_happy_RT_removed.csv',sep='\t')
		# print df_happy_tweets.columns.values
		# print df_happy_tweets.ix[3:10,0:6]
		df_new_happy_tweets = df_happy_tweets.ix[:,0:6]
		df_new_happy_tweets['output']=1
		print df_new_happy_tweets.ix[0:10]

		df_affect = pd.concat([df_new_happy_tweets,df_new_sad_tweets],ignore_index=True)


		print df_affect.ix[0:10]
		print len(df_new_happy_tweets)
		print len(df_new_sad_tweets)
		print len(df_affect)

		npAffect = df_affect.as_matrix()
		print npAffect[0:10,-1]

		y=df_affect['output']
		# sum=0
		# sum_naive = 0
		# sum_knn = 0
		# for x in range(0,5):
        #
		X_train,X_test,y_train,y_test = train_test_split(df_affect.ix[:,0:6],y,test_size=0.2)
		# 	# print "traingin"
		# 	# print X_train
		# 	clf=svm.SVC()
		# 	clf.fit(X_train,y_train)
		# 	prediction = clf.predict(X_test)
        #
		# 	clf_naive = MultinomialNB()
		# 	# clf_naive = GaussianNB()
		# 	clf_naive.fit(X_train,y_train)
		# 	sum_naive += clf_naive.score(X_test,y_test)
		# 	print "naive bayes",
		# 	print clf_naive.score(X_test,y_test)
        #
		# 	neigh = KNeighborsClassifier(n_neighbors=1000)
		# 	neigh.fit(X_train, y_train)
		# 	print "knn",
		# 	print neigh.score(X_test,y_test)
		# 	sum_knn+=neigh.score(X_test,y_test)
        #
        #
		# 	sum+=clf.score(X_test,y_test)
		# 	print "svm",
		# 	print clf.score(X_test,y_test)
        #
		# print "final knn",
		# print sum_knn/5.0
		# print "final naive",
		# print sum_naive/5.0
		# print "final svm",
		# print sum/5.0

		clf=svm.SVC()
		clf.fit(X_train,y_train)
		# prediction = clf.predict(X_test)

		clf_naive = BernoulliNB()
		# clf_naive = GaussianNB()
		clf_naive.fit(X_train,y_train)
		# sum_naive += clf_naive.score(X_test,y_test)
		print "naive bayes",
		print clf_naive.score(X_test,y_test),
		y_pred_naive = clf_naive.predict(X_test)
		scores_naive = precision_recall_fscore_support(y_test, y_pred_naive, average='binary')
		print scores_naive

		neigh = KNeighborsClassifier(n_neighbors=1)
		neigh.fit(X_train, y_train)
		print "knn-1",
		print neigh.score(X_test,y_test)
		y_pred_knn = neigh.predict(X_test)
		scores_knn = precision_recall_fscore_support(y_test, y_pred_knn, average='binary')
		print scores_knn

		neigh = KNeighborsClassifier(n_neighbors=10)
		neigh.fit(X_train, y_train)
		print "knn-10",
		print neigh.score(X_test,y_test)
		y_pred_knn = neigh.predict(X_test)
		scores_knn = precision_recall_fscore_support(y_test, y_pred_knn, average='binary')
		print scores_knn

		neigh = KNeighborsClassifier(n_neighbors=100)
		neigh.fit(X_train, y_train)
		print "knn-100",
		print neigh.score(X_test,y_test)
		y_pred_knn = neigh.predict(X_test)
		scores_knn = precision_recall_fscore_support(y_test, y_pred_knn, average='binary')
		print scores_knn

		neigh = KNeighborsClassifier(n_neighbors=1000)
		neigh.fit(X_train, y_train)
		print "knn-1000",
		print neigh.score(X_test,y_test)
		y_pred_knn = neigh.predict(X_test)
		scores_knn = precision_recall_fscore_support(y_test, y_pred_knn, average='binary')
		print scores_knn

		# sum_knn+=neigh.score(X_test,y_test)


		# sum+=clf.score(X_test,y_test)
		print "svm",
		print clf.score(X_test,y_test)
		y_pred_svm = clf.predict(X_test)
		scores_svm = precision_recall_fscore_support(y_test, y_pred_svm, average='binary')
		print scores_svm


	def classifierDataNGram(self):
		#for affect files
		df_sad_tweets = pd.read_csv('sad_t50.tsv',sep='\t',header=None)

		print df_sad_tweets.ix[0:10,1:]

		print len(df_sad_tweets.columns.values)
		print df_sad_tweets.shape

		print pd.notnull(df_sad_tweets)
		df_sad_tweets['output'] = 0
		df_new_sad_tweets = df_sad_tweets.ix[:,1:]
		nuArr = df_new_sad_tweets.as_matrix()
		print np.any(np.isfinite(nuArr))

		# raw_input("Hi")


		print df_new_sad_tweets.ix[1:10]
		print df_new_sad_tweets.columns.values
		df_new_sad_tweets['output']=0
		print df_new_sad_tweets.columns.values
		print df_new_sad_tweets.ix[1:10]
		# print df_sad_tweets.ix[1:10,0:7]

		df_happy_tweets = pd.read_csv('happy_t50.tsv',sep='\t',header=None)
		# print df_happy_tweets.columns.values
		# print df_happy_tweets.ix[3:10,0:6]
		df_new_happy_tweets = df_happy_tweets.ix[:,1:]
		df_new_happy_tweets['output']=1

		nuArrHappy = df_new_happy_tweets.as_matrix()
		print np.any(np.isfinite(nuArrHappy))

		# raw_input("Hi")

		print df_new_happy_tweets.ix[0:10]

		df_affect = pd.concat([df_new_happy_tweets,df_new_sad_tweets],ignore_index=True)


		print df_affect.ix[0:10]
		print len(df_new_happy_tweets)
		print len(df_new_sad_tweets)
		print len(df_affect)

		npAffect = df_affect.as_matrix()
		print npAffect[0:10,-1]

		y=df_affect['output']
		sum=0
		sum_naive = 0
		sum_knn = 0
		# for x in range(0,5):

		X_train,X_test,y_train,y_test = train_test_split(df_affect.ix[:,0:6],y,test_size=0.3)
		# print "traingin"
		# print X_train
		clf=svm.SVC()
		clf.fit(X_train,y_train)
		# prediction = clf.predict(X_test)

		clf_naive = BernoulliNB()
		# clf_naive = GaussianNB()
		clf_naive.fit(X_train,y_train)
		# sum_naive += clf_naive.score(X_test,y_test)
		print "naive bayes",
		print clf_naive.score(X_test,y_test),
		y_pred_naive = clf_naive.predict(X_test)
		scores_naive = precision_recall_fscore_support(y_test, y_pred_naive, average='binary')
		print scores_naive

		neigh = KNeighborsClassifier(n_neighbors=1)
		neigh.fit(X_train, y_train)
		print "knn-1",
		print neigh.score(X_test,y_test)
		y_pred_knn = neigh.predict(X_test)
		scores_knn = precision_recall_fscore_support(y_test, y_pred_knn, average='binary')
		print scores_knn

		neigh = KNeighborsClassifier(n_neighbors=10)
		neigh.fit(X_train, y_train)
		print "knn-10",
		print neigh.score(X_test,y_test)
		y_pred_knn = neigh.predict(X_test)
		scores_knn = precision_recall_fscore_support(y_test, y_pred_knn, average='binary')
		print scores_knn

		neigh = KNeighborsClassifier(n_neighbors=100)
		neigh.fit(X_train, y_train)
		print "knn-100",
		print neigh.score(X_test,y_test)
		y_pred_knn = neigh.predict(X_test)
		scores_knn = precision_recall_fscore_support(y_test, y_pred_knn, average='binary')
		print scores_knn

		neigh = KNeighborsClassifier(n_neighbors=1000)
		neigh.fit(X_train, y_train)
		print "knn-1000",
		print neigh.score(X_test,y_test)
		y_pred_knn = neigh.predict(X_test)
		scores_knn = precision_recall_fscore_support(y_test, y_pred_knn, average='binary')
		print scores_knn

		# sum_knn+=neigh.score(X_test,y_test)


		# sum+=clf.score(X_test,y_test)
		print "svm",
		print clf.score(X_test,y_test)
		y_pred_svm = clf.predict(X_test)
		scores_svm = precision_recall_fscore_support(y_test, y_pred_svm, average='binary')
		print scores_svm

		# print "final knn",
		# print sum_knn/5.0
		# print "final naive",
		# print sum_naive/5.0
		# print "final svm",
		# print sum/5.0







	def nGram(self,filename,d=500):
		list_unigrams_happy = []
		list_bigrams = []
		list_trigrams = []
    	#HAPPY POSTS
    	# for item in self.new_happyTweets_tokenized:
    	# 	pass
		# print self.data_happy

		# list_unigrams_happy = set([item for sublist in self.new_happyTweets_tokenized for item in sublist])

		# print len(list_unigrams_happy)

		combined_tweet_list_tokenized = [item for sublist in self.new_happyTweets_tokenized for item in sublist]
		combined_tweet_list_tokenized.append(item for sublist in self.new_sadTweets_tokenized for item in sublist)

		# print len(combined_tweet_list_tokenized)
		# print combined_tweet_list_tokenized[0]
		# raw_input("Enter")

		list_unigrams_dist = nltk.FreqDist(combined_tweet_list_tokenized)
		# print len(list_unigrams_dist)
		# print type(list_unigrams_dist)

		list_unigrams_greaterThreshold = []
		list_bigrams_greaterThreshold = []
		list_trigrams_greaterThreshold = []

		# temp_values = list_unigrams_dist.values()
		# temp_values.sort()
		counter = 0
		for k,v in list_unigrams_dist.items():
			# if counter<30:
			# 	counter+=1
			# 	print k,v
			if v>d:
				print k,v
				list_unigrams_greaterThreshold.append(k)

		for item in self.new_happyTweets_tokenized:
			temp_bigrams = nltk.bigrams(item)
			for i in temp_bigrams:
				list_bigrams.append(i)
		for item in self.new_sadTweets_tokenized:
			temp_bigrams=nltk.bigrams(item)
			for i in temp_bigrams:
				list_bigrams.append(i)



		# list_bigrams = nltk.bigrams(combined_tweet_list_tokenized)
		# print type(list_bigrams)
		# print list_bigrams
		# raw_input("Hey There")
		list_bigrams_dist = nltk.FreqDist(list_bigrams)

		for k,v in list_bigrams_dist.items():
			# if counter<30:
			# 	counter+=1
			# 	print k,v
			if v>d:
				print k,v
				list_bigrams_greaterThreshold.append(k)

		for item in self.new_happyTweets_tokenized:
			temp_trigrams = nltk.trigrams(item)
			for i in temp_trigrams:
				list_trigrams.append(i)
		for item in self.new_sadTweets_tokenized:
			temp_trigrams=nltk.trigrams(item)
			for i in temp_trigrams:
				list_trigrams.append(i)
		# list_trigrams = nltk.trigrams(combined_tweet_list_tokenized)


		list_trigrams_dist = nltk.FreqDist(list_trigrams)

		for k,v in list_trigrams_dist.items():
			# if counter<30:
			# 	counter+=1
			# 	print k,v
			if v>d:
				print k,v
				list_trigrams_greaterThreshold.append(k)


		print list_unigrams_greaterThreshold
		print list_bigrams_greaterThreshold
		print list_trigrams_greaterThreshold

		# raw_input("Please Wait")

		list_ngrams = []
		# list_ngrams.append(item for item in list_unigrams_greaterThreshold)
		# list_ngrams.append(item for item in list_bigrams_greaterThreshold )
		# list_ngrams.append(item for item in list_trigrams_greaterThreshold )

		for i in list_unigrams_greaterThreshold:
			list_ngrams.append(i)
		for i in list_bigrams_greaterThreshold:
			list_ngrams.append(i)
		for i in list_trigrams_greaterThreshold:
			list_ngrams.append(i)

		print len(list_ngrams)


		print list_ngrams
		# raw_input("Lets see")

		dict_happy = {'tweets_here01':[]}

		for item in list_ngrams:
			dict_happy[str(item)] = []

		# small_list_tokenized = copy.deepcopy(self.new_happyTweets_tokenized)
		small_list_tokenized = copy.deepcopy(self.new_sadTweets_tokenized)

		print len(small_list_tokenized)
		print small_list_tokenized

		with open(filename, 'w') as outfile:
			# writer = csv.writer(outfile, delimiter="\t")
			for item in range(0,len(small_list_tokenized)):
				# dict_happy['tweets_here01'].append(self.data_happy[item])
				ngram_currentTweet = []
				for i in small_list_tokenized[item]:
					ngram_currentTweet.append(i)
				bigram_currentTweet = nltk.bigrams(small_list_tokenized[item])
				counter_here = 0
				for i in bigram_currentTweet:
					ngram_currentTweet.append(i)
					counter_here+=1
				denominator_bi = counter_here
				trigram_currentTweet = nltk.trigrams(small_list_tokenized[item])
				counter_here = 0
				for i in trigram_currentTweet:
					ngram_currentTweet.append(i)
					counter_here+=1
				denominator_tri=counter_here
				temporary_list = []
				for key in list_ngrams:
					if key in ngram_currentTweet:
						count = 0
						for iterator in ngram_currentTweet:
							if key==iterator:
								count+=1
						print type(key)
						print len(key)
						denominator = 0
						if key in small_list_tokenized[item]:
							denominator=len(small_list_tokenized[item])
						elif len(key)==2:
							# denominator = len(nltk.bigrams(small_list_tokenized[item]))
							denominator = denominator_bi
						elif len(key)==3:
							# denominator = len(nltk.trigrams(small_list_tokenized[item]))
							denominator = denominator_tri
						temporary_list.append(str(float(count)/denominator))
					
					# if key in small_list_tokenized[item] and key != 'tweets_here01':
					# 	temp_join = " ".join(small_list_tokenized[item])

					# 	count=0
					# 	match=re.compile(key)
					# 	count+=len(match.findall(temp_join))
					# 	dict_happy[key].append(float(count)/len(small_list_tokenized[item]))
					# 	temporary_list.append(str(float(count)/len(small_list_tokenized[item])))
					# elif re.compile(key) in nltk.bigrams(small_list_tokenized[item]):
					# 	print key
					# 	raw_input("Bi Gram Match")

					# 	count = 0
					# 	match = re.compile(key) 
					# 	for bigram_temp_here in nltk.bigrams(small_list_tokenized[item]):
					# 		if match == bigram_temp_here:
					# 			count+=1
					# 	temporary_list.append(str(float(count)/len(nltk.bigrams(small_list_tokenized[item]))))
					else:
						# dict_happy[key].append(0)
						temporary_list.append('0')
				temp_string = "\t".join(temporary_list)
				# writer.writerow(self.data_happy[item] + '\t' + temp_string + '\n')
				# outfile.write((self.data_happy[item]).strip() + '\t' + temp_string + '\n')
				outfile.write((self.data_sad[item]).strip() + '\t' + temp_string + '\n')

				# for unigram in small_list_tokenized[item]:
				# 	if unigram in dict_happy.keys():
				# 		print unigram
				# 		# raw_input("Enter Here Please")
				# 		temp_join = " ".join(small_list_tokenized[item])
				# 		count=0
				# 		match=re.compile(unigram)
				# 		count+=len(match.findall(temp_join))
				# 		dict_happy[str(unigram)].append(float(count)/len(small_list_tokenized[item]))
			# bigram_here = nltk.bigrams(small_list_tokenized[item])
			# for bigram_iterator in bigram_here:
			# 	if bigram_iterator in dict_happy.keys():
			# 		temp_join = " ".join(small_list_tokenized[item])
			# 		count=0
			# 		count+=len(bigram_iterator.findall(temp_join))
			# 		dict_happy[str(bigram_iterator)].append(float(count)/len(small_list_tokenized[item]))
			# trigram_here = nltk.trigrams(small_list_tokenized[item])
			# for trigram_iterator in trigram_here:
			# 	if trigram_iterator in dict_happy.keys():
			# 		temp_join = " ".join(small_list_tokenized[item])
			# 		count=0
			# 		count+=len(trigram_iterator.findall(temp_join))
			# 		dict_happy[str(trigram_iterator)].append(float(count)/len(small_list_tokenized[item]))
		print "##################"
		for item in dict_happy:
			print item
			print dict_happy[item]
		print "##################"

		# with open('temp_file.csv', 'wb') as outfile:
		# 	writer = csv.writer(outfile, delimiter="\t")
		# 	# writer.writerow(['tweet','positive_affect','negative_affect','anger','anxiety','sadness','swear'])
		# 	#writer.writerow(liwc.keys())
		# 	# writer.writerows(liwc_count)
		# 	writer.writerow(dict_happy.keys())
		# 	writer.writerows(zip(*dict_happy.values()))	


	def changeLIWC(self,filename):
		finalFile = 'Assignment_II/LIWC_lexicons/' + filename
		temp = open(finalFile).read()
		temp = temp.split('\n')

		liwc_temp = []
		temp_word = 'a'
		with open("{0}/{1}".format('Assignment_II/LIWC_lexicons/', filename), "r") as openFile:
			for item in openFile:
				entry = item.strip()
				if "*" in entry:
					entry = r"\b{0}\b".format(entry.replace('*','.*?'))
				temp_word = "(" + entry + "|"
		temp_word = temp_word[:-1] + ")"
		liwc_temp.append(temp_word)


		return temp

	def findLiwcFrequencies(self):
		dict_affect = {}
		dict_affect['positive_affect'] = self.changeLIWC('positive_affect')
		dict_affect['negative_affect'] = self.changeLIWC('negative_affect')
		dict_affect['anger'] = self.changeLIWC('anger')
		dict_affect['anxiety'] = self.changeLIWC('anxiety')
		dict_affect['sadness'] = self.changeLIWC('sadness')
		dict_affect['swear'] = self.changeLIWC('swear')

		print "REACHED HERE"
		# raw_input("Enter Please")

		liwc_count = []
		final_dict = {'tweets':[],
					  'negative_affect':[],
					  'anxiety': [],
					  'sadness': [],
					  'swear': [],
					  'positive_affect': [],
					  'anger': []}
		counter = 0
		count = 0
		for item in dict_affect:
			# counter = 0
			print "switching"
			# for entry in enumerate(self.new_sadTweets_tokenized):
			for entry in enumerate(self.new_happyTweets_tokenized):
				# print entry
				# liwc_count.append([self.data_happy[entry[0]]])
				

				# final_dict['tweets'].append(self.data_sad[entry[0]])
				final_dict['tweets'].append(self.data_happy[entry[0]])
				

				# temp_tweet = (i.lower() for i in self.new_happyTweets_tokenized[entry[0]])
				# print temp_tweet
				# print self.new_happyTweets_tokenized[entry[0]]
				#raw_input("Enter")
				count = 0
				for x in dict_affect[item]:
					# print x
					match = re.compile(x)
					# print match
					

					# temp_here = " ".join(self.new_sadTweets_tokenized[entry[0]])
					temp_here = " ".join(self.new_happyTweets_tokenized[entry[0]])
					

					count+= len(match.findall(temp_here))	
					# for index in self.new_happyTweets_tokenized[entry[0]]:
					# 	if match == index:
					# 		count += 1
					# 		print "Count Incremented"
				# if len(self.new_sadTweets_tokenized[entry[0]]) !=0:
				if len(self.new_happyTweets_tokenized[entry[0]]) !=0:
					# print count,
					# print len(self.new_happyTweets_tokenized[entry[0]])
					#raw_input("Enter")
					# print entry
					

					# final_dict[item].append(float(count)/len(self.new_sadTweets_tokenized[entry[0]]))
					final_dict[item].append(float(count)/len(self.new_happyTweets_tokenized[entry[0]]))
					

					# liwc_count[len(liwc_count)-1].append(float(count)/len(self.new_happyTweets_tokenized[entry[0]]))
					# counter+=1

		
		with open('tweet_sentiment_happy_RT_removed.csv', 'wb') as outfile:
				writer = csv.writer(outfile, delimiter="\t")
				# writer.writerow(['tweet','positive_affect','negative_affect','anger','anxiety','sadness','swear'])
				#writer.writerow(dict_affect.keys())
				# writer.writerows(liwc_count)
				writer.writerow(final_dict.keys())
				writer.writerows(zip(*final_dict.values()))


if __name__ == '__main__':
	part1Obj = assignment2_Part1()
	# part1Obj.tokenizeDocument()
	# part1Obj.findLiwcFrequencies()
	#part1Obj.nGram('sad_t500.tsv',500)
	#part1Obj.nGram('sad_t50.tsv',50)
	#part1Obj.nGram('sad_t100.tsv',100)

	# part1Obj.classifierDataAffect()
	part1Obj.classifierDataNGram()

	# part1Obj.nGram()
	
