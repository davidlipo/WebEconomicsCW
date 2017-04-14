import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv")

columnsToDrop = [
					'click',
					'weekday',
					'hour',
					'bidid',
					'logtype',
					'userid',
					'useragent',
					'IP',
					#'region',
					#'city',
					#'adexchange',
					#'domain',
					'url',
					'urlid',
					'slotid',
					#'slotwidth',
					#'slotheight',
					'slotvisibility',
					'slotformat',
					'slotprice',
					'creative',
					#'bidprice', #Always remove
					'payprice', #Always remove
					'keypage',
					#'advertiser',
					'usertag'
				]

ycolumn = 'bidprice'

data = data.drop(columnsToDrop, axis=1)

#Replace NaNs with 0
data = data.apply(pd.to_numeric, errors='coerce')
data = data.replace('NaN',0)

#Get x and y for LR
y = data[[ycolumn]]
x = data.drop(ycolumn, axis=1)

#Convert domain to an int
le = preprocessing.LabelEncoder()
le.fit(x['domain'])
x['domain'] = le.transform(x['domain'])

#Run the regression
regr = linear_model.LinearRegression()
regr.fit(x, y)

#Load in the validation set and drop the same columns as with the training set
testOriginal = pd.read_csv("validation.csv")
test = testOriginal.drop(columnsToDrop, axis=1)

#Replace NaNs with 0
test = test.apply(pd.to_numeric, errors='coerce')
test = test.replace('NaN',0)

#Get x and y for LR
testy = test[[ycolumn]]
testclick = testOriginal[['click']]
testx = test.drop(ycolumn, axis=1)

#Convert domain to an int using the same numbers as in the training set
testx['domain'] = le.transform(testx['domain'])

#Run the predictions
predictionOriginal = regr.predict(testx)

#Limit budget
def addToTotal(i, val, click):
  if i > val:
    if total[0] + val < 6250 * 1000:
      total[0] += val
      if click == 1:
        totalClick[0] += val
      return True
  return False

#Bid higher on those that are lower bids
avg = predictionOriginal.mean()
prediction = [(i[0] + 10) if i[0] < avg else (i[0] + 1) for i in predictionOriginal]

#Check the accuracy
total = [0]
totalClick = [0]
success = [k.click for i, j, k in zip(prediction, testy.itertuples(), testclick.itertuples()) if addToTotal(i, j.bidprice, k.click)]
print(sum(success)) #Clicks
print(sum(success) / len(success)) #CTR
print(totalClick[0] / sum(success)) #aCPC
print(total[0] / sum(success) / 1000) #aCPM
print(total[0] / 1000) #Spend
print(" ")

input('Press ENTER to exit')
