# # Part - 1 : File Operations (without pandas)

import csv
with open('Titanic-Dataset.csv','r') as file:
    content = csv.reader(file)
    for i,j in enumerate(content):
        if i <= 5 :
            print(j)

with open('Titanic-Dataset.csv', 'r') as file0:
    contents = csv.reader(file0)
    with open('Titanic-Survivor.csv', 'w') as file1:
        write = csv.writer(file1)
        header = next(contents)
        write.writerow(header)
        for i in contents:
            if i[1] == "1":
                write.writerow(i)
print("Titanic-Survivor.csv created!")

data = []
with open('Titanic-Dataset.csv','r') as f:
    cont = csv.DictReader(f)

    for i in cont:
        data.append(i)
    #print(data)
import json
with open("Titanic-Dataset.json",'w') as f1:
    json.dump(data,f1,indent = 4)
print("Titanic-Dataset.json created!")


# Part - 2 Numpy
import numpy as np
ages = []
fares= []
with open("Titanic-Dataset.csv", "r") as f2:
    read = csv.DictReader(f2)
    for i in read:
        age = i["Age"]
        if age == "":
            ages.append(np.nan)
        else:
            ages.append(float(age))
        fare = i["Fare"]
        if fare == "":
            fares.append(np.nan)
        else:
            fares.append(float(fare))
    age_array = np.array(ages)
    age_mean = np.nanmean(age_array)
    age_array = np.where(np.isnan(age_array), age_mean, age_array)
    mean_val = np.mean(age_array)
    median_val = np.median(age_array)
    std_val = np.std(age_array)
    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Standard Deviation : {std_val}")

  
        
    fare_array = np.array(fares)
    mean_fare = np.nanmean(fare_array)
    fare_array = np.where(np.isnan(fare_array), mean_fare, fare_array)
    fmin = np.min(fare_array)
    fmax = np.max(fare_array)
    fnormalized = (fare_array - fmin) / (fmax - fmin)
    print(fnormalized)



# # Part - 3 : Pandas
import pandas as pd
df = pd.read_csv("Titanic-Dataset.csv")
print("First 5 rows")
print(df.head())
print("Dataset shape: ",df.shape)
print(f"Column names: {df.columns}")
print(f"Data Types:\n {df.dtypes}")
print(f"Total missing values in each column:\n{df.isnull().sum()}")

group = df.groupby("Pclass").agg({"Survived": "mean","Age": "mean","Fare": "mean",})
group.columns = ["Average Survival rate","Age_Mean","Fare_Mean"]
print(group)

df["FamSize"] = df["SibSp"] + df["Parch"]
Fam_Survival = df.groupby("FamSize")["Survived"].mean().sort_values(ascending=False)
Fam_Survival.columns = ["FamilySize","SurvivalRate"]
print(Fam_Survival)

new_df = df[(df["Sex"] == "female")&(df["Age"] >=18)&(df["Age"]<=35)&(df["Pclass"] == 1 )]
new_df.to_csv("Titanic-WomenDataset.csv")
print("Women Dataset Created!")


