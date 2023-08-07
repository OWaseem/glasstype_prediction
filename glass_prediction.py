# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data()

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe):
  glasstype = model.predict([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])  # array([[2]])
  glasstype = glasstype[0]
  if glasstype == 1:
    return "building windows float processed"
  elif glasstype == 2:
    return "building windows non float processed"
  elif glasstype == 3:
    return "vehicle windows float processed"
  elif glasstype == 4:
    return "vehicle windows non float processed"
  elif glasstype == 5:
    return "containers"
  elif glasstype == 6:
    return "tableware"
  elif glasstype == 7:
    return "headlamp"

st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")

if st.sidebar.checkbox("Show Raw Data"):
    st.write("Glass Type Data Set")
    st.dataframe(glass_df)

charts = st.sidebar.multiselect("Charts", ['Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot','Scatter Plot'])
st.set_option('deprecation.showPyplotGlobalUse', False)

if 'Line Chart' in charts:
  st.subheader("Line Chart")
  st.line_chart(glass_df)

if 'Area Chart' in charts:
  st.subheader("Area Chart")
  st.area_chart(glass_df)

if 'Correlation Heatmap' in charts:
  st.subheader("Correlation Heatmap")
  plt.figure(figsize = (15, 5))
  sns.heatmap(glass_df.corr(), annot = True)
  st.pyplot()

if 'Scatter Plot' in charts:
  for feature in X.columns:
    st.subheader(f"Scatter plot between {feature} and GlassType")
    plt.figure(figsize = (12, 6))
    sns.scatterplot(x = feature, y = 'GlassType', data = glass_df)
    st.pyplot()

if 'Count Plot' in charts:
  st.subheader('Count Plot')
  plt.figure(figsize = (15, 5))
  sns.countplot(data = glass_df, x = 'GlassType')
  st.pyplot()

if 'Pie Chart' in charts:
  st.subheader('Pie Chart')
  plt.figure(figsize = (15, 5))
  pie_data = glass_df['GlassType'].value_counts()
  plt.pie(pie_data, labels = pie_data.index, autopct   = '%1.2F%%')
  st.pyplot()

if 'Box Plot' in charts:
  for feature in X.columns:
    st.subheader(f"Box plot between {feature} and GlassType")
    plt.figure(figsize = (15, 5))
    sns.boxplot(data = glass_df, x = feature, y = 'GlassType', orient = 'h')
    st.pyplot()

st.sidebar.subheader("Select your values:")
ri = st.sidebar.slider("Input Ri", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na", float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al", float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider("Input Si", float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider("Input K", float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Input Ba", float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))






classifier = st.sidebar.selectbox("Classifier", ['SupportVectorClassifier', 'LogisticRegression', 'RandomForestClassifier'])


if classifier == 'SupportVectorClassifier':
    st.subheader('SupportVectorMachine')
    c = st.sidebar.number_input('C', 1, 100)
    g = st.sidebar.number_input('Gamma', 1, 100)
    kernel = st.sidebar.radio('kernel', ('linear', 'rbf', 'poly'))
    if st.button("Predict"):
      svc_model = SVC(kernel = kernel, C = c, gamma = g)
      svc_model.fit(X_train, y_train)
      svc_score = svc_model.score(X_train, y_train)
      species_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
      st.write("Species predicted:", species_type)
      st.write("Accuracy score of this model is:", svc_score)

elif classifier == 'RandomForestClassifier':
    estimator = st.sidebar.number_input('estimators', 100, 5000, step = 10)
    max_depth = st.sidebar.number_input('max_depth', 1, 100)
    rfc_model = RandomForestClassifier(n_jobs = -1, n_estimators = estimator, max_depth = max_depth)
    if st.button("Predict"):
      rfc_model.fit(X_train, y_train)
      rfc_score = rfc_model.score(X_train, y_train)
      species_type = prediction(rfc_model, ri, na, mg, al, si, k, ca, ba, fe)
      st.write("Species predicted:", species_type)
      st.write("Accuracy score of this model is:", rfc_score)

elif classifier == 'LogisticRegression':
    max_iteration = st.sidebar.number_input('max_iteration', 1000, 10000, step = 10)
    c = st.sidebar.number_input('c', 1, 100)
    lr_model = LogisticRegression(C = c, max_iter = max_iteration)
    if st.button("Predict"):
      lr_model.fit(X_train, y_train)
      lr_score = lr_model.score(X_train, y_train)
      species_type = prediction(lr_model, ri, na, mg, al, si, k, ca, ba, fe)
      st.write("Species predicted:", species_type)
      st.write("Accuracy score of this model is:", lr_score) 