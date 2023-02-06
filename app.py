import streamlit as st
import seaborn as sns
import pandas as pd
import plotly as px
import numpy as np
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

# DataFrame
df = sns.load_dataset('tips')
df = df.dropna()   # drop null values
df = df.drop_duplicates()

# headers
st.write("""
# Predictions on 'Tips' Dataset
#### In this app, we'll perform EDA on 'Tips' dataset, we'll then see the data before and after wrangling the data, then we'll predict 'Tip'
""")

st.write('**First five columns:**', df.head())

# EDA
st.write('## **EDA of Dataframe:**')
pr = ProfileReport(df, explorative=True)
st_profile_report(pr)


# Wrangling
st.write('## Before and After Wrangling')

# total_bill column (boxplots)
st.write('## Outliers')
col1, col2 = st.columns(2)

fig1 = px.boxplot_frame(df, y='total_bill', width=400)
col1.write("#### 'Total Bill' Before")
col1.plotly_chart(fig1)

fig2 = px.boxplot_frame(df[df['total_bill'] <= 38], y='total_bill', width=400)
col2.write("#### 'Total Bill' After")
col2.plotly_chart(fig2)

# tips columns (boxplots)
col3, col4 = st.columns(2)

fig3 = px.boxplot_frame(df, y='tip', width=400)
col3.write("#### 'Tip' Before")
col3.plotly_chart(fig3)

fig4 = px.boxplot_frame(df[df['tip'] <= 5], y='tip', width=400)
col4.write("#### 'Tip' After")
col4.plotly_chart(fig4)

# total_bill column (histogram)
st.write('## Distributions')
col5, col6 = st.columns(2)

fig5 = px.hist_frame(df['total_bill'], width=400)
col5.write("#### 'Total Bill' Before")
col5.plotly_chart(fig5)

fig6 = px.hist_frame(np.log(df['total_bill']), width=400)
col6.write("#### 'Total Bill' After")
col6.plotly_chart(fig6)

# tips column (histogram)
st.write('## Distributions')
col7, col8 = st.columns(2)

fig7 = px.hist_frame(df['tip'], width=400)
col7.write("#### 'Tip' Before")
col7.plotly_chart(fig7)

fig8 = px.hist_frame(np.log(df['tip']), width=400)
col8.write("#### 'Tip' After")
col8.plotly_chart(fig8)



# ML part
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


side = st.sidebar
side.write("# User Input Parameters")
model = side.selectbox('**Select ML algorithm**', ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'Support Vector Regressor', 'KNN'])
st.write('# Predictions using different ML models')


# features and labels
X = df[['total_bill']]
y = df['tip']


# parameters for Linear Regression
if model == 'Linear Regression' or model == 'Decision Tree Regressor' or model == 'Support Vector Regressor':
    def user_input_features():
        total_bill = side.slider('Total_bill', 20, 45, 30)
        data = {'total_bill' : total_bill}
        features = pd.DataFrame(data, index=[0])
        return features
    df = user_input_features()
    st.write("## User Input Parameters")
    st.write(df)

    if model == 'Linear Regression':
        reg = LinearRegression()
        reg.fit(X,y)
        prediction = reg.predict(df)
        y = y.reindex(df['total_bill'].index)


        st.subheader("Predicted 'Tip'")
        st.write(prediction)

    elif model == 'Decision Tree Regressor':
        reg = DecisionTreeRegressor()
        reg.fit(X,y)
        prediction = reg.predict(df)
        y = y.reindex(df['total_bill'].index)


        st.subheader("Predicted 'Tip'")
        st.write(prediction)

    elif model == 'Support Vector Regressor':
        reg = SVR()
        reg.fit(X,y)
        prediction = reg.predict(df)
        y = y.reindex(df['total_bill'].index)


        st.subheader("Predicted 'Tip'")
        st.write(prediction)

elif model == 'Random Forest Regressor':
    def user_input_features():
        global n_estimators, max_depth
        total_bill = side.slider('Total_bill', 20, 45, 30)
        n_estimators = side.slider('How many trees you want ?', 50, 300, 100)
        max_depth = side.slider('How much depth you want ?', 5, 50, 20)
        data = {'total_bill' : total_bill,
                'n_estimators' : n_estimators,
                'max_depth' : max_depth}
        features = pd.DataFrame(data, index=[0])
        return features
    df = user_input_features()
    st.write("## User Input Parameters")
    st.write(df)

    # model
    reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    reg.fit(X,y)
    prediction = reg.predict(df[['total_bill']])
    y = y.reindex(df['total_bill'].index)


    st.subheader("Predicted 'Tip'")
    st.write(prediction)

elif model == 'KNN':
    def user_input_features():
        global neighbours, data
        total_bill = side.slider('Total_bill', 20, 45, 30)
        neighbours = side.slider('How many neighbours yo want ?', 3,7,2)
        data = {'total_bill' : total_bill,
                'neighbours' : neighbours}
        features = pd.DataFrame(data, index=[0])
        return features
    df = user_input_features()
    st.write("## User Input Parameters")
    st.write(df)

    # model
    reg = KNeighborsRegressor(n_neighbors=neighbours)
    reg.fit(X,y)
    prediction = reg.predict(df[['total_bill']])
    y = y.reindex(df['total_bill'].index)

    st.subheader("Predicted 'Tip'")
    st.write(prediction)


# metrics evaluation
st.write("## Metrics Evaluation")

st.subheader("Mean Absolute Error of the model is: ")
st.write(mean_absolute_error(y, prediction))
st.subheader("Mean Squared Error of the model is: ")
st.write(mean_squared_error(y, prediction))
st.subheader("Root Mean Squared Error of the model is: ")
st.write(mean_squared_error(y, prediction, squared=False))