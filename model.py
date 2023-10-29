import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load the CSV file
data = pd.read_csv("./patientinput.csv")

# Data preprocessing
# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split the data into features and target
X = data.drop(columns=['Opioid Risk Level'])
y = data['Opioid Risk Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

def predictRisk(patientData):
	patientData = patientData[X.columns]
	prediction = clf.predict(patientData)[0]
	return {
		"predictedRisk": int(prediction)
    }

# new_example = pd.DataFrame({
#     'Age': [30],
#     'Gender': [1],
#     'Race/Ethnicity': [1],
#     'Family History of Substance Abuse': [0],
#     'Personal History of Substance Abuse': [0],
#     'History of Mental Health Conditions': [0],
#     'Chronic Pain Conditions': [1],
#     'Prescribed Medications': [2],
#     'Dosage and Frequency of Opioid Medication Use': [10],
#     'Duration of Opioid Medication Use': [10],
#     'History of Overdose or Hospitalization Due to Opioid Use': [1],
#     'Social Support Network': [0],
#     'Employment Status': [1],
#     'Education Level': [3]
# })
# print("Predicted Opioid Risk Level:", predictRisk(new_example))