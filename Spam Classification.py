#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset=pd.read_csv('SMSSpamCollection',sep='\t',header=None)
dataset.head()


# In[3]:


dataset.columns=["label","message"]


# In[4]:


dataset.head()


# In[5]:


dataset.label.value_counts()
#dataset seems tobe unbalanced


# In[ ]:





# In[6]:


#Data Cleaning and Text preprocessing


# In[7]:


import re
import nltk
nltk.download('stopwords')


# In[8]:


from nltk.corpus import stopwords


# In[9]:


stopwords.words('english')


# In[10]:


stopwords_list=['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'nor',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]


# In[11]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[12]:


corpus=[]
for i in range(0,len(dataset)):
    review=re.sub('[^a-zA-Z0-9]',' ',dataset['message'][i]) #Substitute with blank
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords_list]
    review=' '.join(review)
    corpus.append(review)


# In[13]:


corpus


# In[ ]:





# In[14]:


#independent and dependent features
y=pd.get_dummies(dataset['label'],drop_first=True)


# In[15]:


y


# In[16]:


#Train and Test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(corpus,y,test_size=0.2,random_state=0)


# In[17]:


x_train


# In[18]:


y_train.head()


# In[ ]:





# In[19]:


#Creating and BOW 


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train).toarray()


# In[21]:


x_train


# In[22]:


x_test=cv.transform(x_test).toarray()


# In[23]:


x_test


# In[24]:


x_train.shape
#here we can se that 4457 sentences and 6485 vocabulary words


# In[25]:


y_train.shape


# In[ ]:





# In[26]:


#Now we can manage vacabulary words which are most frequent 


# In[27]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(corpus,y,test_size=0.2,random_state=0)


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
x_train=cv.fit_transform(x_train).toarray()


# In[29]:


x_train


# In[30]:


x_train.shape
#here we can see that vacabulary words changes to 2500


# In[31]:


cv.vocabulary_
#{'word':index_no.}


# In[ ]:





# In[32]:


#Now we apply ngrams with max_features 


# In[33]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(corpus,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500,ngram_range=(1,2))
x_train=cv.fit_transform(x_train).toarray()


# In[34]:


x_train


# In[35]:


x_train.shape
#here we get same no. of features given in max_features


# In[36]:


cv.vocabulary_


# In[37]:


x_test=cv.transform(x_test).toarray()


# In[38]:


x_test


# In[ ]:





# In[ ]:





# In[39]:


#Machine Learning Algorithems


# In[40]:


# 1]Randomforest


# In[60]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier().fit(x_train,y_train)


# In[61]:


y_pred=classifier.predict(x_test)


# In[62]:


y_pred


# In[63]:


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# In[64]:


accuracy_score(y_test,y_pred)


# In[65]:


confusion_matrix(y_test,y_pred)


# In[66]:


print(classification_report(y_test,y_pred))


# In[ ]:




msg = input("Enter Message: ")
msgInput = cv.transform([msg])
predict = classifier.predict(msgInput)
if(predict[0]==0):
    print("------------------------MESSAGE-SENT-[CHECK-SPAM-FOLDER]---------------------------")
else:
    print("---------------------------MESSAGE-SENT-[CHECK-INBOX]------------------------------")

# In[ ]:





# In[ ]:





# In[48]:


# 2]xgboost


# In[68]:


import xgboost as xgb


# In[69]:


model = xgb.XGBClassifier()
model.fit(x_train, y_train)
print(model)


# In[70]:


expected_y  = y_test
predicted_y = model.predict(x_test)
predicted_y


# In[71]:


from sklearn import metrics
print(metrics.classification_report(expected_y, predicted_y))
print(metrics.confusion_matrix(expected_y, predicted_y))


# In[ ]:





# In[72]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)


# In[73]:


pred=spam_detect_model.predict(x_test)


# In[74]:


pred


# In[75]:


expected_y  = y_test
predicted_y = spam_detect_model.predict(x_test)
predicted_y


# In[76]:


from sklearn import metrics
print(metrics.classification_report(expected_y, predicted_y))
print(metrics.confusion_matrix(expected_y, predicted_y))


# In[ ]:


#!pip install flask


# In[79]:


import pickle
pickle.dump(spam_detect_model,open('model.pkl','wb'))


# In[81]:


pickle_model=pickle.load(open('model.pkl','rb'))
#Batch input
pickle_model.predict(x_test)


# In[ ]:





# In[92]:


#Check it with random message
msg = input("Enter Message: ")
msgInput = cv.transform([msg])
predict = spam_detect_model.predict(msgInput)
if(predict[0]==0):
    print("------------------------MESSAGE-SENT-[CHECK-SPAM-FOLDER]---------------------------")
else:
    print("---------------------------MESSAGE-SENT-[CHECK-INBOX]------------------------------")


# In[93]:


predict


# In[ ]:




