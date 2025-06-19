from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# metadata 
print(bank_marketing.metadata) 
  
# variable information 
print(bank_marketing.variables) 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Lets check if the data has loaded correctly first
print("Getting a few rows of X")
print(X.head())
print()
print("Getting a few rows of y")
print(y.head())
print()

#Lets check the shapes of our Data Frames
print("Shape of bank marketing data features is: ",X.shape)
print("Shape of bank marketing data targets is: ",y.shape)
print()

#For easier Analysis I wauld prefer to combine both into one Data Frame
df = pd.concat([X,y], axis=1)
print(df.head())
print()

#Lets code categorial Variables first

#I will replace the missing values first
df['job'].fillna('unknown', inplace=True)
df['education'].fillna('unknown', inplace=True)
df['poutcome'].fillna('unknown', inplace=True)
df['contact'].fillna('unknown', inplace=True)

#Lets binary code the colums
binary_cols = ['default', 'housing', 'loan', 'contact', 'y']
df[binary_cols] = df[binary_cols].apply(lambda col: col.map({'yes': 1, 'no': 0}))

df['marital'] = df['marital'].map({'married':1, 'single':0, 'unknown':0})

#Hot encoded the jobs
df['job'] = df['job'].apply(lambda col: False if col == 'unknown' or col.lower() == 'unemployed' else True)
df['education'] = df['education'].apply(lambda col: False if col == 'unknown' or col.lower() == 'uneducated' else True)
df['poutcome'] = df['poutcome'].apply(lambda col: False if col == 'unknown' or col.lower() == 'failure' else True)


#Ordinal Encoding For day of the week and month
day_mapping = {'mon':0, 'tue':1, 'wed':2, 'thu':3, 'fri':4, 'sat':5, 'sun':6}
df['day_of_week'] = df['day_of_week'].map(day_mapping)

month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df['month'] = df['month'].map(month_mapping)

#Now that we have encoded all varibles we can handle the missinng values and perform scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Handling values that are missing
numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

#Lets Normalize the data
df[numerical_cols] = MinMaxScaler().fit_transform(df[numerical_cols])

#I believe encoding of pdays and previous is not required in the provided data set but I will be doing it here anyways just as preactice
#Lets segregate into bins
df['pdays_bins'] = pd.cut(df['pdays'], bins=[-2,0,100,300,1000], labels=['Not Contacted', 'Recently Contacted', 'Moderatly Contacted', 'Long Ago'])
df['previous_category'] = pd.cut(df['previous'], bins=[-1, 0, 1, 5, 100], labels=['None', 'Few', 'Moderate', 'Many'])

#Lets check the Balance-CallDuration Ratio
df['balance_duration_ratio'] = df['balance']/(df['duration'] +1) # +1 to stop divion by 0

#Lets also study contacts per day
df['contacts_per_day'] = df['campaign']/(df['pdays'] +1)  # +1 to stop divion by 0

#Handle missing valuesbefore SMOTE to ensure no NaN values are present
df['pdays_bins'] = df['pdays_bins'].astype(str).factorize()[0]
df['previous_category'] = df['previous_category'].astype(str).factorize()[0]

#SMOTE
from imblearn.over_sampling import SMOTE
df_copy = df.copy()
S1 = df_copy.drop(columns = ['y','pdays_bins', 'previous_category', 'contact', 'day_of_week']) #The part of data frame without the targeted volumn 'y' in this case
S2 = df_copy['y']                                                                              #The part of data frame with just the targeted volumn 'y' in this case

#To remove any Nan is any categoty in S1
S1.fillna(S1.median(), inplace=True)

S1_resampled,S2_resampled = SMOTE().fit_resample(S1,S2)

print("Resampled feature shape: ", S1_resampled.shape)
print("Resampled target shape: ", S2_resampled.shape)
print()
S1_resampled = pd.DataFrame(S1_resampled, columns=S1.columns)
resampled_df = pd.concat([S1_resampled, pd.Series(S2_resampled, name='y')], axis=1)
print("Resampled df")
print(resampled_df.head)

#Now we will visualize things and bring out actual valuable information from this DataFrame
#Exploratry Data Analysis (EDA)

#Demographic Analysis: Distribution of 'age', 'job', and 'matrital' status
print()
print("Demographic Analysis: Distribution of 'age'")
plt.figure(figsize=(10,6))
sns.histplot(df['age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

print()
print("Demographic Analysis: Distribution of 'job'")
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='job')
plt.title('Job Distibution')
plt.xticks(rotation=45)
plt.show()

print()
print("Demographic Analysis: Distribution of 'marital'")
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='marital')
plt.title('Marital Status Distribution')
plt.show()

#Correlationn Analysis: Heatmap of numerical columns
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#Campaign Effectiveness: Call duration vs outcome
plt.figure(figsize=(10,6))
sns.boxplot(data = df, x='y',y='duration')
plt.title('Call Duration By Campaign Outcome')
plt.xlabel('Campaign Outcome (0 = No, 1=Yes)')
plt.ylabel('Sucess Rate')
plt.show()

#Trend of sucess based on month and week day
month_sucess_rate = df.groupby('month')['y'].mean()
month_sucess_rate.plot(kind='bar' , color='skyblue')
plt.title('Sucess Rate By Month')
plt.xlabel('Month')
plt.ylabel('Sucess Rate')
plt.show()

day_sucess_rate = df.groupby('day_of_week')['y'].mean()
month_sucess_rate.plot(kind='bar' , color='orange')
plt.title('Sucess Rate By Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Sucess Rate')
plt.show()

#Balance vs Duration Scatter Plot
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='balance', y='duration', hue='y', alpha=0.7)
plt.title('Balance vs Call Duration')
plt.xlabel('Balance')
plt.ylabel('Call Duration in seconds')
plt.legend(title='Campaign Outcome')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


X_train, X_test, y_train, y_test = train_test_split(S1_resampled, S2_resampled, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

print(classification_report(y_test, y_predicted))

cm = confusion_matrix(y_test, y_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.show()
