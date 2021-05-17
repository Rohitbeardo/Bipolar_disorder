
import pandas as pd
import sklearn


data=pd.read_csv("C:\Users\saiki\OneDrive\Desktop\Bipolar\rohit_bipolar\input.csv")
data.head()



# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(data.loc[:,data.columns != "episode"], 
#                                                     data["episode"], test_size=0.2)

# """Decision tree"""

# from sklearn import tree

# dt =tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=15)
# dt.fit(X_train, y_train)

# """Random Forest"""

# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier(n_jobs=-1)
# rf.fit(X_train, y_train)

# """Linear Regression"""

# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression()
# lr.fit(X_train, y_train)

# """Function to print predicitions"""

# def check_state(classifier, data):
#     prediction = classifier.predict(data)[0]
#     if prediction == 'D':
#         diagnosis = "The patient could be tending towards a Depression episode"
#     elif prediction == 'M':
#         diagnosis = "The patient could be tending towards a Mania episode"
#     else:
#         diagnosis = "The patient is in a normal state"
#     return diagnosis

# from random import randint
# import numpy as np

# """Prediction using random numbers"""

# n = 0
# while n < 5:
#     variables = {}
#     variables['mood'] = randint(-3, 3)
#     variables['motivation'] = randint(-3, 3)
#     variables['attention'] = randint(0, 4)
#     variables['irritability'] = randint(0, 4)
#     variables['anxiety'] = randint(0, 4)
#     variables['sleep_quality'] = randint(0, 4)
#     variables['caffeine'] = randint(0, 250)
#     variables['active_time'] = randint(200, 1800)
    
#     test = np.array([variables['mood'], variables['motivation'], variables['attention'],
#                 variables['irritability'], variables['anxiety'], variables['sleep_quality'],
#                 variables['caffeine'], variables['active_time']])
    
#     print ("--------------------------------------------")
#     print ("PATIENT ", n + 1)
#     print ("--------------------------------------------")
#     print ("Mood: ", variables['mood'])
#     print ("Motivation: ", variables['motivation'])
#     print ("Attention: ", variables['attention'])
#     print ("Irritability: ", variables['irritability'])
#     print ("Anxiety: ", variables['anxiety'])
#     print ("Sleep quality: ", variables['sleep_quality'])
#     print ("Caffeine: ", variables['caffeine'])
#     print ("Active time: ", variables['active_time'])
    
#     test = test.reshape(1, -1)
#     print ("--------------------------------------------")
#     print ("PREDICTIONS")
#     print ("--------------------------------------------")
#     print ("- Decision Tree prediction: ", check_state(dt, test))
#     print ("- Random Forest prediction: ", check_state(rf, test))
#     print ("- Logistic Regression prediction: ", check_state(lr, test))
    
#     print ("\n")
    
#     n += 1

# """Predicting manually"""

# n = 0
# while n < 1:
#     variables = {}
#     variables['mood'] = 0
#     variables['motivation'] = 0
#     variables['attention'] = 0
#     variables['irritability'] = 0
#     variables['anxiety'] = 0
#     variables['sleep_quality'] = 2
#     variables['caffeine'] = 130
#     variables['active_time'] = 1709
    
#     test = np.array([variables['mood'], variables['motivation'], variables['attention'],
#                 variables['irritability'], variables['anxiety'], variables['sleep_quality'],
#                 variables['caffeine'], variables['active_time']])
    
#     print ("--------------------------------------------")
#     print ("PATIENT ", n + 1)
#     print ("--------------------------------------------")
#     print ("Mood: ", variables['mood'])
#     print ("Motivation: ", variables['motivation'])
#     print ("Attention: ", variables['attention'])
#     print ("Irritability: ", variables['irritability'])
#     print ("Anxiety: ", variables['anxiety'])
#     print ("Sleep quality: ", variables['sleep_quality'])
#     print ("Caffeine: ", variables['caffeine'])
#     print ("Active time: ", variables['active_time'])
    
#     test = test.reshape(1, -1)
#     print ("--------------------------------------------")
#     print ("PREDICTIONS")
#     print ("--------------------------------------------")
#     print ("- Decision Tree prediction: ", check_state(dt, test))
#     print ("- Random Forest prediction: ", check_state(rf, test))
#     print ("- Logistic Regression prediction: ", check_state(lr, test))
    
#     print ("\n")
    
#     n += 1

# pip install anvil-uplink

# import anvil.server

# anvil.server.connect("V2CJ354GIPITMG3XHCJPUJ32-GYFDVRVXGFQJML2K")

# def output_lable(n):
#   if n == 'D':
#     return"The patient could be tending towards a Depression episode"
#   elif n == 'M':
#       return "The patient could be tending towards a Mania episode"
#   else:
#         return "The patient is in a normal state"

# pred_DT = dt.predict(test)

# test

# import anvil.media

# @anvil.server.callable
# def predict(mood,motivation,attention,irritability,anxiety,sleep_quality,caffiene,active_time):
#    test = np.array([mood,motivation,attention,irritability,anxiety,sleep_quality,caffiene,active_time])
#    test = test.reshape(1, -1)
#    pred_LR = lr.predict(test)
#    pred_DT = dt.predict(test)
#    str1= output_lable(pred_LR[0])
#    str2= output_lable(pred_DT[0])
# return ( "LR prediction : "+str1+ "\n\nDT prediction : "+str2)





# def predict(mood,motivation,attention,irritability,anxiety,sleep_quality,caffiene,active_time):
#    test = np.array([mood,motivation,attention,irritability,anxiety,sleep_quality,caffiene,active_time])
#    test = test.reshape(1, -1)
#    print(check_state(dt,test))





# import anvil.media 
# @anvil.server.callable
# def output_lable(n):
#     if n == 0:
#         return "FAKE"
#     elif n == 1:
#         return "REAL"
    
# def manual_testing(news):
#     testing_news = {"text":[news]}
#     new_def_test = pd.DataFrame(testing_news)
#     new_def_test["text"] = new_def_test["text"].apply(wordopt) 
#     new_x_test = new_def_test["text"]
#     new_xv_test = vectorization.transform(new_x_test)
#     pred_LR = LR.predict(new_xv_test)
#     pred_DT = DT.predict(new_xv_test)
#     pred_GBC = GBC.predict(new_xv_test)
#     pred_RFC = RFC.predict(new_xv_test)
#     str1= output_lable(pred_LR[0])
#     str2= output_lable(pred_DT[0])
#     str3=output_lable(pred_GBC[0])
#     str4=output_lable(pred_RFC[0])
#     return "LR prediction : "+str1+ "\n\nDT prediction : "+str2+"\n\nGBC Prediction : "+str3

# def predict(mood,motivation,attention,irritability,anxiety,sleep_quality,caffiene,active_time):
#    test = np.array([mood,motivation,attention,irritability,anxiety,sleep_quality,caffiene,active_time])
#    test = test.reshape(1, -1)
#    print(check_state(dt,test))

# import anvil.media

# @anvil.server.callable
# def predict(mood,motivation,attention,irritability,anxiety,sleep_quality,caffiene,active_time):
#    test = np.array([mood,motivation,attention,irritability,anxiety,sleep_quality,caffiene,active_time])
#    test = test.reshape(1, -1)
#    return check_state(dt,test)