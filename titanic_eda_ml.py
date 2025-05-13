import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Titanic dataset
@st.cache_data
def load_data():
    df = pd.read_csv("titanic dataset.csv")
    return df

df = load_data()

# Title of the Streamlit app
st.title("Titanic Dataset - EDA & ML Model")

# Display the first few rows of the dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Display basic stats
st.subheader("Basic Statistics")
st.write(df.describe())

# Display dataframe info using buffer
st.subheader("DataFrame Info")
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

# Display correlation heatmap
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Show missing values
st.subheader("Missing Data")
st.write(df.isnull().sum())

# Show distribution of 'Age'
st.subheader("Age Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True, color='blue', ax=ax)
st.pyplot(fig)

# Show the survival rate by 'Pclass'
st.subheader("Survival Rate by Pclass")
pclass_survival = df.groupby('Pclass')['Survived'].mean()
st.bar_chart(pclass_survival)

# Prepare data for Machine Learning
df['Age'].fillna(df['Age'].mean(), inplace=True)
df_cleaned = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_cleaned['Embarked'].fillna(df_cleaned['Embarked'].mode()[0], inplace=True)
df_cleaned = pd.get_dummies(df_cleaned, drop_first=True)

# Split features and target
X = df_cleaned.drop('Survived', axis=1)
y = df_cleaned['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Model Performance
st.subheader("Model Performance")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Interactive Prediction
st.subheader("Make a Prediction")
age = st.number_input("Age", min_value=0, max_value=100, value=30)
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
embarked = st.selectbox("Embarked", ['C', 'Q', 'S'])

# Construct input vector
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'Sex_male': [1 if sex == 'male' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0]
})

# Add any missing columns from training set
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns
input_data = input_data[X.columns]

# Scale input
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

# Output result
if prediction[0] == 1:
    st.success("🎉 Predicted Survival: **Survived**")
else:
    st.error("❌ Predicted Survival: **Not Survived**")
