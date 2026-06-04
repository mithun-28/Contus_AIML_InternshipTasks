import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score)
from imblearn.over_sampling import SMOTE

df = pd.read_csv("processed_fraud_data.csv")
print("Dataset Shape:", df.shape)
X = df.drop("is_fraud",axis=1)
y = df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
print("\nAfter SMOTE")
print(pd.Series(y_train_smote).value_counts())

rf_model = RandomForestClassifier(n_estimators=300,max_depth=15,min_samples_split=5,min_samples_leaf=2,random_state=42,n_jobs=-1)
rf_model.fit(X_train_smote,y_train_smote)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]

print("RANDOM FOREST RESULTS")
print("\nClassification Report")
print(classification_report(y_test,y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("\nPrecision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1 Score:",f1_score(y_test,y_pred))
print("ROC AUC:",roc_auc_score(y_test,y_prob))


joblib.dump(rf_model,"Randomforest.joblib")
