import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor

train_data = pd.read_excel(r"Data_Train.xlsx")
pd.set_option('display.max_columns', None)
# print(train_data.head())
# print(train_data.info())

# exploring the data
# print(train_data['Duration'])
# print(train_data['Duration'].value_counts())

# dropping the NaN value
train_data.dropna(inplace=True)
# print(train_data.isnull().sum())

# exploratory data analysis
# print(train_data.head())

# Extracting journey day and month in separate variable from Date_of_Journey
train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'], format="%d/%m/%Y").dt.day
train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'], format="%d/%m/%Y").dt.month
# print(train_data.info())
# Dropping the Date_of_Journey column as it is of no use. axis = 1 means checks the column name
train_data.drop(['Date_of_Journey'], axis=1, inplace=True)
# print(train_data.info())

# Extracting hours and minutes from Dep_Time
train_data['Dep_hours'] = pd.to_datetime(train_data['Dep_Time']).dt.hour
train_data['Dep_min'] = pd.to_datetime(train_data['Dep_Time']).dt.minute
# print(train_data.info())
# Dropping Dep_Time column
train_data.drop(['Dep_Time'], axis=1, inplace=True)
# print(train_data.info())

# Extracting hours and minutes from Arrival_Time
train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
train_data['Arrival_min'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute
# Dropping Arrival_Time column
train_data.drop(['Arrival_Time'], axis=1, inplace=True)
# print(train_data.info())

# Extracting Duration hours and minutes from Duration
# Duration is in Format eg. 12hr 56m so we have to altered the data inorder to extract it
duration = list(train_data['Duration'])  # taking all value from Duration and keeping it in list

for i in range(len(duration)):  # this for loop is run till the length of duration
    if len(duration[i].split()) != 2:  # checking if duration contains only hours or minutes
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + " 0m"  # adding minute
        else:
            duration[i] = "0h " + duration[i]  # adding hour

duration_hour = []
duration_min = []
# Extracting Duration hours and minutes from Duration

for i in range(len(duration)):
    duration_hour.append(int(duration[i].split(sep='h')[0]))
    duration_min.append(int(duration[i].split(sep='m')[0].split()[-1]))

# Adding duration hours and minutes  in train_data
train_data['Duration_hour'] = duration_hour
train_data['Duration_min'] = duration_min
# print(train_data.info())

# Dropping duration
train_data.drop('Duration', axis=1, inplace=True)
# print(train_data.info())

# Handling categorical data
# print(train_data['Airline'].value_counts())

# sns.catplot(y="Price", x="Airline", data=train_data.sort_values('Price', ascending=False), kind="boxen", height=6, aspect=3)
# plt.show()

# As Airline is Nominal Categorical data so i performed OneHotEncoding
Airline = train_data[['Airline']]
# OneHotEncoding
Airline = pd.get_dummies(Airline, drop_first=True)
# print(Airline.head())

# Working with Source column
# print(train_data['Source'].value_counts())
# sns.catplot(y="Price", x="Source", data=train_data.sort_values('Price', ascending=False), kind="boxen", height=6, aspect=3)
Source = train_data[["Source"]]
Source = pd.get_dummies(Source, drop_first=True)
# print(Source.head())

# Working with Destination column
# print(train_data['Destination'].value_counts())
# sns.catplot(y="Price", x="Destination", data=train_data.sort_values('Price', ascending=False), kind="boxen", height=6, aspect=3)
# plt.show()
Destination = train_data[['Destination']]
Destination = pd.get_dummies(Destination, drop_first=True)
# print(Destination.head())

# working on Route and Additional Info column
# print(train_data["Route"])
# print(train_data["Additional_Info"].head())

# since Additional_info contains almost 80% no_info(no data) so dropping the column
# Route and total_stops are related to each other so also dropping route column
train_data.drop(['Additional_Info', 'Route'], axis=1, inplace=True)

# working with Total_Stops
# print(train_data["Total_Stops"].value_counts())
# as this is case of Ordinal Categorical type so LabelEncoder is used
# Values are assigned to with corresponding keys
train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)
# print(train_data.head())

# concatenate dataframe / combining all the processed data into data frame which is train_data
data_train = pd.concat([train_data, Airline, Source, Destination], axis=1)
# print(data_train.head())

# as we combined the processed data now the drop the dropping the columns which are not required
data_train.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)
# print(data_train.head())

# print(data_train.shape)


# Test Set
test_data = pd.read_excel(r"Test_set.xlsx")
# print(test_data.head())

# dropping all NaN in test_data
test_data.dropna(inplace=True)

# Exploratory data analysis
# Extracting journey day and month in separate variable from Date_of_Journey
test_data['Journey_day'] = pd.to_datetime(test_data['Date_of_Journey'], format="%d/%m/%Y").dt.day
test_data['Journey_Month'] = pd.to_datetime(test_data['Date_of_Journey'], format="%d/%m/%Y").dt.month
# print(test_data.head())
# Dropping Date_of_Journey column
test_data.drop('Date_of_Journey', axis=1, inplace=True)
# print(test_data)

# Extracting hours and minutes from Dep_Time
test_data['Dep_hour'] = pd.to_datetime(test_data['Dep_Time']).dt.hour
test_data['Dep_min'] = pd.to_datetime(test_data['Dep_Time']).dt.minute
# print(test_data.head())
# Dropping Dep_Time column
test_data.drop('Dep_Time', axis=1, inplace=True)
# print(test_data.head())

# Extracting hours and minutes from Arrival_Time
test_data['Arrival_hour'] = pd.to_datetime(test_data['Arrival_Time']).dt.hour
test_data['Arrival_minute'] = pd.to_datetime(test_data['Arrival_Time']).dt.minute
# print(test_data.head())
# Dropping Arrival_Time Column
test_data.drop('Arrival_Time', axis=1, inplace=True)
# print(test_data.head())

# Extracting hours and minutes from Duration
Duration = list(test_data["Duration"])
Duration_hour = []
Duration_min = []

for i in range(len(Duration)):
    if len(Duration[i].split()) != 2:  # checking if the Duration has both hours and minutes
        if 'h' in Duration[i]:
            Duration[i] = Duration[i].strip() + " 0m"  # adding minutes
        else:
            Duration[i] = "0h " + Duration[i]  # adding hours

for i in range(len(Duration)):
    Duration_hour.append(int(Duration[i].split(sep='h')[0]))
    Duration_min.append(int(Duration[i].split(sep='m')[0].split()[-1]))

# adding Duration_hours and Duration_minute to data frame
test_data['Duration_hour'] = Duration_hour
test_data['Duration_min'] = Duration_min

# Dropping Duration column
test_data.drop("Duration", axis=1, inplace=True)
# print(test_data.head())

# Handling categorical data
# print(test_data['Airline'].value_counts())
# working with Airline column
airline = test_data['Airline']
airline = pd.get_dummies(test_data['Airline'], drop_first=True)
# print(airline.head())

# Working with Source
source = test_data['Source']
source = pd.get_dummies(test_data["Source"], drop_first=True)
# print(source.head())

# working with Destination column
destination = test_data['Destination']
destination = pd.get_dummies(test_data['Destination'], drop_first=True)
# print(destination.head())

# Working with route and additional info
# print(test_data['Route'].head())

# Additional info not contain much data so dropping that column
# Route and total_stops are related to each other so also dropping route column
test_data.drop(['Additional_Info', 'Route'], axis=1, inplace=True)
# print(test_data.head())

# Assigning corresponding values to Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)
# print(test_data['Total_Stops'].head())

# combining / concatenating all processed data to test data frame
data_test = pd.concat([test_data, airline, source, destination], axis=1)
# print(data_test.head())

# dropping the column
data_test.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)
# print(data_test.head())


# Feature Selection
# print(data_train.shape)
# print(data_train.columns)
# X = independent values i.e Price is not included
X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hours',
                       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hour',
                       'Duration_min', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
                       'Airline_Jet Airways', 'Airline_Jet Airways Business',
                       'Airline_Multiple carriers',
                       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
                       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
                       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
                       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
                       'Destination_Kolkata', 'Destination_New Delhi']]
# print(X.head())

# Y = Dependent value i.e Price is included Price is in position 1
Y = data_train.iloc[:, 1]
# print(Y.head())

# Finding correlation between Independent and Dependent attributes
plt.figure(figsize=(18, 18))
# sns.heatmap(train_data.corr(), annot=True, cmap='RdYlGn')
# plt.show()

# Getting Important features using ExtraTreeRegressor
ETree_Reg = ExtraTreesRegressor()
ETree_Reg.fit(X, Y)
print(ETree_Reg.feature_importances_)
