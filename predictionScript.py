from sklearn import svm
import csv
import copy
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

class assignment2_Part1:
	def __init__(self):
		temp = open('Assignment_II/data/pos_examples_happy.txt','r')
		print temp
		self.data_happy = []
		for item in temp:
			result = re.sub(r"http\S+", "", item.encode('ascii','ignore'))
			self.data_happy.append(result)

		temp = open('Assignment_II/data/neg_examples_sad.txt','r')
		print temp
		self.data_sad = []
		for item in temp:
			result = re.sub(r"http\S+", "", item.encode('ascii','ignore'))
			self.data_sad.append(result)

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

	def LiwcOccurences(self,filename):
		newFileName = 'Assignment_II/LIWC_lexicons/' + filename
		temp = open(newFileName).read()
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
		liwc = {}
		liwc['positive_affect'] = self.LiwcOccurences('positive_affect')
		liwc['negative_affect'] = self.LiwcOccurences('negative_affect')
		liwc['anger'] = self.LiwcOccurences('anger')
		liwc['anxiety'] = self.LiwcOccurences('anxiety')
		liwc['sadness'] = self.LiwcOccurences('sadness')
		liwc['swear'] = self.LiwcOccurences('swear')

		print "REACHED HERE"
		raw_input("Enter Please")

		liwc_count = []
		# liwc_lex_count = {'tweets':[], 'negative_affect':[], 'anxiety': [], 'sadness': [], 'swear': [], 'positive_affect': [], 'anger': []}
		counter = 0
		count = 0
		for item in liwc:
			# counter = 0
			print "switching"
			for entry in enumerate(self.new_happyTweets_tokenized):
				# print entry
				liwc_count.append([self.data_happy[entry[0]]])
				

				# liwc_lex_count['tweets'].append(self.data_happy[entry[0]])
				

				# temp_tweet = (i.lower() for i in self.new_happyTweets_tokenized[entry[0]])
				# print temp_tweet
				# print self.new_happyTweets_tokenized[entry[0]]
				#raw_input("Enter")
				count = 0
				for x in liwc[item]:
					# print x
					match = re.compile(x)
					# print match
					temp_here = " ".join(self.new_happyTweets_tokenized[entry[0]])
					count+= len(match.findall(temp_here))	
					# for index in self.new_happyTweets_tokenized[entry[0]]:
					# 	if match == index:
					# 		count += 1
					# 		print "Count Incremented"
				if len(self.new_happyTweets_tokenized[entry[0]]) !=0:
					# print count,
					# print len(self.new_happyTweets_tokenized[entry[0]])
					#raw_input("Enter")
					# print entry
					# liwc_lex_count[item].append(float(count)/len(self.new_happyTweets_tokenized[entry[0]]))
					liwc_count[len(liwc_count)-1].append(float(count)/len(self.new_happyTweets_tokenized[entry[0]]))
					# counter+=1

		with open('tweet_sentiment_sad.csv', 'wb') as outfile:
			writer = csv.writer(outfile, delimiter="\t")
			# writer.writerow(['tweet','positive_affect','negative_affect','anger','anxiety','sadness','swear'])
			writer.writerow(liwc.keys())
			writer.writerows(liwc_count)
			# writer.writerow(liwc_lex_count.keys())
			# writer.writerows(zip(*liwc_lex_count.values()))



		# print liwc['positive_affect_list']






		# if stopWordInstruction==True:
		# 	pass
		# 	for item in happyTweetsTokenized:
		# 		temp = []
		# 		temp += (word for word in item if word.lower() not in stopwords_mine)
		# 		new_happyTweets_tokenized.append(temp)

  #       else:
  #       	new_happyTweets_tokenized=copy.deepcopy(happyTweetsTokenized)
        # else:
        # 	new_happyTweets_tokenized=copy.deepcopy(happyTweetsTokenized)
    def gramCalculation(self):
    	list_unigrams_happy = []
    	list_bigrams_happy = []
    	list_trigrams_happy = []
    	#HAPPY POSTS
    	for item in self.new_happyTweets_tokenized:




    	#SAD POSTS
    	list_unigrams_sad = []
    	list_bigrams_sad = []
    	list_trigrams_sad = []
              
            

        


if __name__ == '__main__':
	part1Obj = assignment2_Part1()
	part1Obj.tokenizeDocument()
	part1Obj.findLiwcFrequencies()
	
