from sklearn import svm
import csv
import copy
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

class assignment2_Part1:
	def __init__(self):
		temp = open('Assignment_II/data/pos_examples_happy.txt','r')
		print temp
		self.data_happy = []
		for item in temp:
			self.data_happy.append(item.encode('ascii','ignore'))

		temp = open('Assignment_II/data/neg_examples_sad.txt','r')
		print temp
		self.data_sad = []
		for item in temp:
			self.data_sad.append(item)

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
			temp_tokenized = ([item1.encode('ascii','ignore') for item1 in temp_tokenized if item1.encode('ascii','ignore')!= 'RT'])
			temp_tokenized = " ".join(temp_tokenized)
			temp_tokenized = regExpTokenizer.tokenize(temp_tokenized)

			happyTweetsTokenized.append([item1.encode('ascii','ignore') for item1 in temp_tokenized if item1.encode('ascii','ignore')!= 'RT'])
		raw_input("Press Enter")
		# for item in happyTweetsTokenized:
		# 	print item
		new_happyTweets_tokenized = []
		if stopWordInstruction==True:
			for item in happyTweetsTokenized:
				temp = []
				temp += (word for word in item if word.lower() not in stopwords_mine)
				new_happyTweets_tokenized.append(temp)
		else:
			new_happyTweets_tokenized=copy.deepcopy(happyTweetsTokenized)

		for item in new_happyTweets_tokenized:
			print item
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
	
