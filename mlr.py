import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page configuration
st.set_page_config(page_title="Multiple Linear Regression", layout="wide")

# Sidebar - Logo and Authors
st.sidebar.image(
    "https://www.asean-competition.org/file/post_image/LCyh3I_post_MyCC.jpg",  # Replace with your image URL or local path
    #caption="MyCC",
    use_container_width=True
)
#st.sidebar.image(
#    "https://brand.umpsa.edu.my/images/2024/02/29/umpsa-bangunan__1764x719.png",  # Replace with your image URL or local path
#    caption="UMPSA",
#    use_container_width=True
#)

st.sidebar.header("Developers:")
Developers = [
    "Ku Muhammad Naim Ku Khalif",
    "Izzat"
    # Add additional author names here
]
  
for developer in Developers:
    st.sidebar.write(f"- {developer}")

# Main Title
st.title("Multiple Linear Regression Framework with Prediction")

# Sidebar - Upload data and select features
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    features = st.sidebar.multiselect("Select independent variables (X)", df.columns)
    target = st.sidebar.selectbox("Select dependent variable (y)", df.columns)

    if features and target and target not in features:
        X = df[features]
        y = df[target]

        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Performance")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")
        st.write("Coefficients:", dict(zip(features, model.coef_)))
        st.write("Intercept:", model.intercept_)

        # Actual vs Predicted plot
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Prediction input
        st.subheader("Make a Prediction")
        input_data = []
        for feature in features:
            val = st.number_input(
                f"Input value for {feature}",
                value=float(np.mean(df[feature]))
            )
            input_data.append(val)

        if st.button("Predict"):
            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            st.success(f"Predicted {target}: {prediction:.3f}")

    else:
        st.info("Please select at least one feature and a target variable (target should not be in features).")
else:
    st.info("Upload a CSV file to start.")
