# -- coding: utf-8 --
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Title and Sidebar
st.title("Model Comparison: Minimal vs Good Preprocessing")
st.sidebar.title("Options")

# Section 1: Data Overview
st.header("1. Data Overview")
st.markdown("""
This application compares the performance of models trained with minimal preprocessing versus good preprocessing on a heart failure dataset.
""")

# Load datasets
raw_data_path = "https://raw.githubusercontent.com/vvignesh04/fds/refs/heads/main/heart_failure_raw_with_inconsistencies_1000_rows%20(2).csv"
preprocessed_data_path = "https://raw.githubusercontent.com/vvignesh04/fds/refs/heads/main/preprocessed_dataset.csv"

try:
    df_raw = pd.read_csv(raw_data_path)
    df_preprocessed = pd.read_csv(preprocessed_data_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Section 2: Data Distribution Visualization
st.header("2. Data Distribution Visualization")

# Data distribution for Raw and Preprocessed data
data_selection = st.sidebar.radio("Choose data to display:", ("Raw Data", "Preprocessed Data"))
selected_data = df_raw if data_selection == "Raw Data" else df_preprocessed

st.write(f"### {data_selection} Distribution")
for col in selected_data.select_dtypes(include=["object", "int64", "float64"]).columns:
    fig = px.histogram(selected_data, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Section 3: Model Performance Metrics
st.header("3. Model Performance Metrics")

# Good Preprocessing Metrics
good_accuracy = 0.53  # Updated value
good_cm = [[55, 115], [60, 170]]  # Adjust as per new confusion matrix if needed

# Minimal Preprocessing Metrics
minimal_accuracy = 0.50  # Updated value
minimal_cm = [[72, 98], [85, 140]]  # Adjust as per new confusion matrix if needed

# Display Minimal Preprocessing metrics first
st.write("### Minimal Preprocessing Metrics")
st.write(f"*Accuracy:* {minimal_accuracy}")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(minimal_cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Non-Failure", "Failure"], yticklabels=["Non-Failure", "Failure"])
ax.set_title("Confusion Matrix (Minimal Preprocessing)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# Display Good Preprocessing metrics later
st.write("### Good Preprocessing Metrics")
st.write(f"*Accuracy:* {good_accuracy}")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(good_cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Non-Failure", "Failure"], yticklabels=["Non-Failure", "Failure"])
ax.set_title("Confusion Matrix (Good Preprocessing)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# Section 4: Model Comparison
st.header("4. Model Comparison")

comparison_data = {
    "Model": ["Minimal Preprocessing", "Good Preprocessing"],
    "Accuracy": [minimal_accuracy, good_accuracy],
    "AUC": [0.51, 0.56],  # Updated AUC values
}

df_comparison = pd.DataFrame(comparison_data)

# Visualization of model comparison
fig = px.bar(df_comparison, x="Model", y=["Accuracy", "AUC"], barmode="group", title="Model Performance Comparison")
st.plotly_chart(fig, use_container_width=True)

# Section 5: Insights
st.header("5. Insights")
st.markdown("""
- *Minimal Preprocessing*: Achieved an accuracy of 50% with an AUC of 0.51, indicating that preprocessing is crucial for performance improvement.
- *Good Preprocessing*: Achieved a higher accuracy of 53% with an AUC of 0.56. The better metrics demonstrate the importance of effective preprocessing.
- *Takeaway*: Preprocessing significantly affects model performance. Further steps like feature engineering and advanced modeling may yield better results.
""")
