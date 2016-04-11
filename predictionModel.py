import csv
import pandas as pd
from sklearn import svm

class modelTraining:
	def __init__(self,filePath):
		self.filePath = filePath
		self.df_temp = pd.read_csv("PatientData.csv")
		self.listB = [s.lower() for s in self.df_temp.columns.values]
		self.df_temp.columns = self.listB
		self.featureList = []

	def featureInput(self):
		print "available features are:",
		for index,feature in enumerate(self.df_temp.columns.values):
			print index,
			print ".",
			print feature
		userInput = raw_input("Please input the parameters in double quotes, and comma separated. Keep the names same as displayed.")

		list_parameters = [x.lower() for x in userInput.split(',')]
		featureVerification(list_parameters)

	def featureVerification(self,inputList):
		featureFlag = True
		for userInput_feature in inputList:
			if userInput_feature not in self.df_temp.columns.values:
				featureFlag = False
				print "Feature not found:",
				print userInput_feature

		if featureFlag == False:
			print "please input the features again"
			featureInput()
		else:
			print "Features Recorded"
			self.featureList = inputList
	
	def trainModel(self,inputFeatureList = self.featureList):
		df1 = pd.DataFrame(self.df_temp, columns=featureList)
		self.model = svm.SVC()
		self.model.fit(df1,self.df_temp[:,-1])

	def predictValues(inputData):
		self.model.predict(inputData)

	def otherParts():
		#print df_temp
		print df_temp.columns.values
		listA = df_temp.columns.values
		
		print listA
		print listB

		
		print df_temp.columns.values

		print "************"
		#print "Please input the parameters in double quotes, and comma separated"
		
		print list_parameters
		print type(list_parameters)
		







if __name__=="__main__":
	print "Please input the path for data file"
	path = raw_input()
	object1 = modelTraining(path)
	object1.featureInput()
	# object1.predictValues(**)

