import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load the csv files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')


# Prepare training data
# ----------------------
trainingData = []

## retrieve baseline measure for current patient
for id in train['Patient'].unique():
    currPatient = train[train['Patient'] == id]
    startingMeasure = list(currPatient.iloc[0, :].values)
    ## for current patient, retrieve their subsequent measures in later weeks
    for i, week in enumerate(currPatient['Weeks'].iloc[1:]):
        fvc = currPatient.iloc[i, 2]
        trainDataPoint = startingMeasure + [week, fvc]
        trainingData.append(trainDataPoint)
trainingData = pd.DataFrame(trainingData)

trainingData.columns = ['PatientID', 'start_week', 'start_FVC', 'start_Percent', 'Age', 'Sex', 'SmokingStatus'] + ['curr_week', 'curr_FVC']
## create new column that indicates the change in week from baseline
trainingData['delta_week'] = trainingData['curr_week'] - trainingData['start_week']     
trainingData.drop(columns = ['start_Percent', 'curr_week', 'start_week'], inplace = True)

## encode cateogrical features: Sex, SmokingStatus
le = LabelEncoder()
trainingData['Sex'] = le.fit_transform(trainingData['Sex'])
trainingData['SmokingStatus'] = le.fit_transform(trainingData['SmokingStatus'])


# Prepare testing data
# ---------------------
testData = []
patient_Week = np.array(list(sample['Patient_Week'].apply(lambda x: x.split('_')).values))

## retrieve baseline measure for current patient
for p in np.unique(patient_Week[:, 0]):
    currPatient = test[test['Patient'] == p]
    firstMeasure = list(currPatient.iloc[0, :].values)
    ## for current patient, retrieve their subsequent measures in later weeks (no FVC since we need to predict that)
    for week in patient_Week[patient_Week[:, 0] == p, 1]:
        testDataPoint = firstMeasure + [week]
        testData.append(testDataPoint)
testData = pd.DataFrame(testData)
testData.columns = ['PatientID', 'start_week', 'start_FVC', 'start_Percent', 'Age', 'Sex', 'SmokingStatus'] + ['curr_week']
## create new column that indicates the change in week from baseline
testData['delta_week'] = testData['curr_week'].map(int) - testData['start_week']
testData.drop(columns = ['start_Percent', 'start_week'], inplace = True)

## encode categorical features: Sex, SmokingStatus
testData['Sex'] = le.fit_transform(testData['Sex'])
testData['SmokingStatus'] = le.fit_transform(testData['SmokingStatus'])


# train model and make predictions
model = LinearRegression()
trainX = trainingData.drop(columns = ['PatientID', 'curr_FVC'])
trainY = trainingData['curr_FVC']
testX = testData.drop(columns = ['PatientID', 'curr_week'])
model.fit(trainX, trainY)
prediction = model.predict(testX)


# append predictions to test dataframe
testData['predicted_FVC'] = prediction

# main execution
if __name__ == "__main__":
    print(testData.head(20))
