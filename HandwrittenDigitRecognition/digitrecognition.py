import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score)


df = pd.read_csv("mnist_train_small.csv",header=None)
columns = ['label'] + [f'pixel{i}' for i in range(1, 785)]
df.columns = columns
print("Shape:", df.shape)
print(df.head())

print("\nDataset Info")
print(df.info())
print("\nMissing Values")
print(df.isnull().sum().sum())
print("\nClass Distribution")
print(df['label'].value_counts().sort_index())

plt.figure(figsize=(8,5))
sns.countplot(x='label', data=df)
plt.title("Digit Distribution")
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(10,5))
for i, ax in enumerate(axes.flat):
    image = df.iloc[i, 1:].values.reshape(28,28)
    label = df.iloc[i, 0]
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')
plt.tight_layout()
plt.show()

X = df.drop('label', axis=1)
y = df['label']
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
model = RandomForestClassifier(n_estimators=200,max_depth=None,random_state=42,n_jobs=-1)

model.fit(X_train, y_train)
print("\nTraining Completed!")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score( y_test,y_pred,average='weighted')
recall = recall_score(y_test,y_pred,average='weighted')

f1 = f1_score(y_test,y_pred,average='weighted')

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

joblib.dump(model, "mnist_digit_classifier.pkl")
print("\nModel Saved Successfully!")