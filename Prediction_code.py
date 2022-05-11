//import pandas as pd
from calendar import month_abbr
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

from google.colab import files
uploaded = files.upload()
import io
dataset = pd.read_csv(io.BytesIO(uploaded['breast-cancer.csv']))

# DATA CLEANING
# Correcting data errors in column 'tumor-size' and 'inv nodes'
dict = {}
for month in range(1,13):
    dict[month_abbr[month]] = month

column_errors = ['tumor-size', 'inv-nodes']
for x in column_errors:
    newlist = []
    for entry in dataset[x]:
        if len(entry)>5:
            for month in dict.keys():
                if month in entry:
                    newlist.append(entry.replace(month, str(dict[month])))
        else:
            newlist.append(entry)
    dataset[x] = newlist

# Addressing missing values
MissingValues = SimpleImputer(missing_values = '?', strategy = 'most_frequent')
MissingValues = MissingValues.fit(dataset)
dataset2 = pd.DataFrame(MissingValues.transform(dataset), columns = dataset.columns)

# Summary statistics 
column_list = dataset2.columns.tolist()
for column in column_list:
  frequency = dataset2[column].value_counts()
  dictionary = frequency.to_dict()
  x = []
  y = []
  for i in dictionary:
    x.append(i)
    y.append(dictionary[i])
  plt.bar(x,y)
  plt.xlabel(column)
  plt.ylabel('Frequency')
  plt.show()
column_list.remove('Class')
for column in column_list:
  cg = pd.crosstab(dataset2['Class'], dataset2[column])
  cg.plot.bar(stacked = True)
  plt.ylabel('Frequency')
  plt.show()

# FEATURE SELECTION
from scipy.stats import chi2_contingency
for feature in dataset2.columns:
    table = feature + 'contingencyTable'
    table = pd.crosstab(dataset2[feature], dataset2['Class'], margins=False)
    # print(table)
    # print(chi2_contingency(table))
"""Chi square tests indicated that Class was only dependent on inv_nodes, node_caps, irradiat, and deg-malig
at statistically significant levels. Hence only these features were used in the model."""  

#Encoding categorical variables
inv_nodes_dict = {'0-2':1, '03-5':2, '06-8':3, '09-11':4, '12-14':5, '15-17':6, '24-26':7}
node_caps_dict = {'yes':1, 'no':0}
irradiat_dict = {'yes':1, 'no':0}
class_dict = {'recurrence-events':1, 'no-recurrence-events':0}

list1 = [inv_nodes_dict, node_caps_dict, irradiat_dict, class_dict]
list2 = ['temp_inv_nodes', 'temp_nodecaps', 'temp_irrad', 'temp_class']
list3 = ['inv-nodes', 'node-caps', 'irradiat', 'Class']

class LabellingCode():
    def __init__(self, column, col_dict, new_column, dataframe):
        ser = pd.Series(data=dataframe[column])
        dataframe[new_column]= ser.map(col_dict)

for i in range(0, len(list3)):
    LabellingCode(list3[i], list1[i], list2[i], dataset2)

dataset2 = dataset2.drop(['age', 'menopause', 'tumor-size', 'breast', 'breast-quad', 'inv-nodes',
                          'node-caps', 'irradiat', 'Class'], axis=1)
dataset2['deg-malig'] = dataset2['deg-malig'].astype('object').astype('int64')

X = dataset2.iloc[:, :-1]
y = dataset2.iloc[:, -1]

# Addressing imbalanced classification in Class (recurrence)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_smote, y_smote = sm.fit_resample(X, y)

# Training the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.25, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# Building the artificial neural network (ANN)
import tensorflow as tf 
import keras 
from keras.models import Sequential 
from keras.layers import Dense, Dropout

# initialise the ANN
classifier = Sequential()
# classifier.add (Dropout(0.1, input_dim = 4)) 
"""No benefit of dropout (to prevent overfitting) on the model accuracy was found"""
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the ANN
# classifier.fit(X_train, y_train, epochs = 200, batch_size = 30)
history = classifier.fit(X_train, y_train, epochs=100, batch_size=30, validation_data=(X_val, y_val))

# predicting and evaluating the ANN
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
"""The model produced the best recall score for recurrence of cases (approx. 70%),
 which was deemed an important evaluation measure for the predictive task at hand"""

# ROC graph 
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_pred)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
plt.title(f'AUC: {auc}')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
plt.show()

# Model accuracy graph 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Model loss graph 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Validation'], loc='upper left') 
plt.show()
