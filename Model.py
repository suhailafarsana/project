import pandas as pd
import pickle

data= pd.read_csv('C:/Users/Wahu_Buzz/Downloads/TestAPP1/Final_data.csv')
     
from sklearn.model_selection import train_test_split

# lets split the target data from the train data

y = data['Y']
X = data.drop(['Y'], axis = 1)
x_train, x_test, y_train, y_test= train_test_split(X,y, random_state= 30,test_size= 0.2)

from sklearn.preprocessing import OrdinalEncoder
order = [['Home','No Urgent Place', 'Work'],['Kid(s)','Alone','Partner','Friend(s)'],['Rainy','Snowy','Sunny'],['30','55','80'],['7AM','10AM','2PM','6PM','10PM'],
         ['Bar','Restaurant(20-50)','Coffee House','Restaurant(<20)','Carry out & Take away'],['2h','1d'],['Female','Male'],
         ['Widowed','Divorced','Married partner','Unmarried partner','Single'],[0,1],
         ['Some High School','High School Graduate','Some college - no degree','Associates degree','Bachelors degree','Graduate degree (Masters or Doctorate)'],
         ['Healthcare Support','Construction & Extraction','Healthcare Practitioners & Technical','Protective Service','Architecture & Engineering','Production Occupations','Student','Office & Administrative Support','Transportation & Material Moving','Building & Grounds Cleaning & Maintenance','Management','Food Preparation & Serving Related','Life Physical Social Science','Business & Financial','Computer & Mathematical','Sales & Related','Personal Care & Service','Unemployed','Farming Fishing & Forestry','Installation Maintenance & Repair','Education&Training&Library','Arts Design Entertainment Sports & Media','Community & Social Services','Legal','Retired'],
         ['Less than $12500','$12500 - $24999','$25000 - $37499','$37500 - $49999','$50000 - $62499','$62500 - $74999','$75000 - $87499','$87500 - $99999','$100000 or More'],
         ['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],
        [0,1],['Below 21','21-30','31-40','41-50','50 & Above'],['within 15min','Between 15-25 min','morethan25min']]
Ordinal_enc = OrdinalEncoder(categories=order)
Encoder = Ordinal_enc.fit(x_train)
pickle.dump(Encoder, open("C:/Users/Wahu_Buzz/Downloads/TestAPP1/Encoder.pkl", "wb"))
Data_Ord_enc = Encoder.transform(x_train)
Data_Ord_enc = pd.DataFrame(Data_Ord_enc,columns=(x_train.columns.values))
Test_Data_Ord_enc = Encoder.transform(x_test)
Test_Data_Ord_enc = pd.DataFrame(Test_Data_Ord_enc,columns=(x_test.columns.values))

from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score    
from catboost import CatBoostClassifier
Cb_model= CatBoostClassifier()
# Checking with the Decision tree Classifier model
Cb_model.fit(Data_Ord_enc, y_train)
y_pred7= Cb_model.predict(Test_Data_Ord_enc)
print('Accuracy Score is:', accuracy_score(y_test, y_pred7))
print('Recall Score is:', recall_score(y_test, y_pred7))
print('Precision Score:', precision_score(y_test, y_pred7))
print('F1 score is:', f1_score(y_test, y_pred7))
pickle.dump(Cb_model, open("C:/Users/Wahu_Buzz/Downloads/TestAPP1/model.pkl", "wb"))
