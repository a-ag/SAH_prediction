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
				temp += (word for word in item if word.lower() not in stopwords_mine)
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
				temp += (word for word in item if word.lower() not in stopwords_mine)
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
		with open("{0}/{1}".format('LIWC_lexicons', input), "r") as openFile:
			for line in openFile:
				word = line.strip()
				if "*" in word:
					word = r"\b{0}\b".format(word.replace('*','.*?'))
				lex_str = "(" + word + "|"
		lex_str = lex_str[:-1] + ")"
		liwc_temp.append(lex_str)


		return temp

	def findLiwcFrequencies(self):
		liwc = {}
		liwc['positive_affect_list'] = self.LiwcOccurences('positive_affect')
		liwc['negative_affect_list'] = self.LiwcOccurences('negative_affect')
		liwc['anger_list'] = self.LiwcOccurences('anger')
		liwc['anxiety_list'] = self.LiwcOccurences('anxiety')
		liwc['sadness_list'] = self.LiwcOccurences('sadness')
		liwc['swear_list'] = self.LiwcOccurences('swear')

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
              
            

        


if __name__ == '__main__':
	part1Obj = assignment2_Part1()
	part1Obj.tokenizeDocument()
	part1Obj.findLiwcFrequencies()
	
