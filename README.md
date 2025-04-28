<H3>Name: Kurapati Vishnu Vardhan Reddy</H3>
<H3>Register No: 212223040103</H3>
<H3>EX. NO.1</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
# Import necessary libraries
import pandas as pd                  
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split         

# Load the dataset from a CSV file
data = pd.read_csv("Churn_Modelling.csv")

# Display the entire DataFrame (not needed if using Jupyter Notebook as it auto-displays the last variable)
data

# Show the first 5 rows of the dataset
data.head()

# Extract all columns except the last one into feature matrix X
X = data.iloc[:, :-1].values
X

# Extract the last column into target vector y
y = data.iloc[:, -1].values
y

# Check for any missing values in the dataset
data.isnull().sum()

# Check for duplicate rows
data.duplicated()

# Get summary statistics for numerical columns
data.describe()

# Drop non-numeric and potentially irrelevant columns before scaling
# 'Surname', 'Geography', and 'Gender' are categorical and dropped for simplicity
data = data.drop(['Surname', 'Geography', 'Gender'], axis=1)
data.head()

# Apply Min-Max scaling to normalize the feature values between 0 and 1
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(data))  # Create a new DataFrame with scaled data
print(df1)

# Split the original (non-scaled) data into training and test sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Print the training feature set and its length
print(X_train)
print(len(X_train))

# Print the test feature set and its length
print(X_test)
print(len(X_test))


```


## OUTPUT:
### Dataset Preview
<img width="779" alt="Screenshot 2025-04-06 at 7 58 07 PM" src="https://github.com/user-attachments/assets/c3dc1681-b85b-475e-9022-f7e2b1f4237a" />


### Feature Matrix (X Values)
<img width="468" alt="Screenshot 2025-04-06 at 7 58 23 PM" src="https://github.com/user-attachments/assets/ae9a45f4-2203-4ec6-89b4-6d5b84254316" />


### Target Vector (Y Values)
<img width="327" alt="Screenshot 2025-04-06 at 7 58 35 PM" src="https://github.com/user-attachments/assets/c2ce54e8-2a89-46d5-90aa-9b13a1441edd" />


### Missing Values Check
<img width="376" alt="Screenshot 2025-04-06 at 7 58 48 PM" src="https://github.com/user-attachments/assets/716e7df8-3f3f-4bcb-b9cb-eb973b888127" />


### Duplicate Records Check
<img width="247" alt="Screenshot 2025-04-06 at 7 59 00 PM" src="https://github.com/user-attachments/assets/6553e6ea-6792-4f84-a242-2322603ccecc" />


### Dataset Statistical Summary
<img width="777" alt="Screenshot 2025-04-06 at 7 59 12 PM" src="https://github.com/user-attachments/assets/31a073fc-c2c8-4a0e-a096-d9a9dd971809" />


### Normalized Dataset
<img width="763" alt="Screenshot 2025-04-06 at 7 59 47 PM" src="https://github.com/user-attachments/assets/2a799f21-0da6-4e04-a584-4214999782af" />


### Training Data (X_train)
<img width="534" alt="Screenshot 2025-04-06 at 8 00 16 PM" src="https://github.com/user-attachments/assets/534a1cda-ac62-4399-a9ba-1fc2d7f73f3e" />


### Testing Data (X_test)
<img width="530" alt="Screenshot 2025-04-06 at 8 00 24 PM" src="https://github.com/user-attachments/assets/48b7c3f2-a875-47d2-b005-2c58398118b1" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


