import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

def load_dataset():
    # Load your weather dataset from a CSV file
    df = pd.read_csv('C:/Users/valer/Documents/Machine Learning/Practicas/Bases de datos/DATASET E-COMERCE/Sales Transaction v.4a.csv')
    df = df[df['Quantity'] > 0]
    columns = df.select_dtypes(include=['number']).columns
    for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['CustomerOrderCount'] = df.groupby('CustomerNo')['TransactionNo'].transform('count')
    df['ProductSaleFrequency'] = df.groupby('ProductNo')['Quantity'].transform('count')   
    df['Revenue'] = df['Price']*df['Quantity']
    df = df.sample(frac=0.1, random_state=42)
    df.reset_index(drop=True, inplace=True)
    return df

def get_model(algorithm):
    if algorithm == 'Linear Regression':
        model = LinearRegression()
    elif algorithm == 'Support Vector Regressor':
        model = SVR(C=25)
    return model

def build_pipeline(algorithm='Linear Regression'):
    numerical_cols = ["Month", "IsWeekend", "ProductSaleFrequency","CustomerOrderCount",'Price', 'Quantity']
    categorical_cols = ["Country"]
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()), 
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols), 
        ("cat", cat_pipeline, categorical_cols), 
    ])
    model = get_model(algorithm)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline

def train_model(df, algorithm='Linear Regression'):

    df["product_sale_frequency_cat"] = pd.cut(df["ProductSaleFrequency"],
                               bins=[0.,400, 800.,np.inf],
                               labels=[1, 2,3])
    df["revenue_cat"] = pd.cut(df["Revenue"],
                               bins=[0.,50, 100.,np.inf],
                               labels=[1, 2,3])
    df['stratify_col']= df["IsWeekend"].astype(str)+"_"+df["Month"].astype(str)+"_"+df["product_sale_frequency_cat"].astype(str)+"_"+df["revenue_cat"].astype(str)
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["stratify_col"]):
        strat_train_set = df.loc[train_index]  #Conjunto de entrenamiento
        strat_test_set = df.loc[test_index] # Conjunto de prueba

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("product_sale_frequency_cat", axis=1, inplace=True)
        set_.drop("stratify_col", axis=1, inplace=True)
        set_.drop("revenue_cat", axis=1, inplace=True)

    y_train = strat_train_set["Revenue"].copy()
    X_train = strat_train_set.drop("Revenue", axis=1)

    y_test = strat_test_set["Revenue"].copy()
    X_test = strat_test_set.drop("Revenue", axis=1)

    pipeline = build_pipeline(algorithm)
    pipeline.fit(X_train, y_train)
    evaluate_model(pipeline, X_test, y_test)

    # Save the model to a pickle file
    with open(f'{algorithm.lower().replace(" ", "_")}_model.pkl', 'wb') as model_file:
        pickle.dump(pipeline, model_file)

    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    # Evaluate the model
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)

    # Calculate Adjusted R-squared
    n = len(y_test)
    k = X_test.shape[1]
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

    # Display metrics
    st.subheader('Metricas de Evaluación:')
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
    st.write(f'R-squared (R²): {r2:.4f}')
    st.write(f'Adjusted R-squared: {adj_r2:.4f}')
    st.write(f'Explained Variance Score: {explained_var:.4f}')

    # Display scatter plot and residual plot
    fig, ax1 = plt.subplots(1, figsize=(8, 4))
    ax1.scatter(y_test, y_pred, alpha=0.7)
    ax1.set_title('Real vs. Predicho')
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Predicho')
    st.pyplot(fig)

# Function to predict 
def predict_revenue(model, month, is_weekend, product_sale_frequency, customer_order_count, country, price, quantity):
    input_data = pd.DataFrame([{
        "Month": month,
        "IsWeekend": is_weekend,
        "ProductSaleFrequency": product_sale_frequency,
        "CustomerOrderCount": customer_order_count,
        "Price": price,
        "Quantity": quantity,
        "Country": country
    }])
    predicted_revenue = model.predict(input_data)
    return predicted_revenue[0]

# Main Streamlit app
def main():
    st.title('Predicción de Ventas')

    # Load weather dataset
    df = load_dataset()

    # Select regression algorithm
    algorithm = st.sidebar.selectbox('Seleccionar algortimo',
                                     ['Linear Regression', 'Support Vector Regressor'])

    # Train the model
    model = train_model(df, algorithm)

    ###
    month_names = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4,
        'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8,
        'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
    }

    is_weekend_names = {'No': 0, 'Si': 1}

    # Streamlit UI
    st.sidebar.header('Datos de Entrada del Usuario')
    month = month_names[st.sidebar.selectbox("Selecciona un mes",list(month_names.keys()))]
    is_weekend = is_weekend_names[st.sidebar.selectbox('Selecciona si es fin de semana', list(is_weekend_names.keys()))]
    country = st.sidebar.selectbox('Selecciona el pais', df['Country'].unique().tolist())
    product_sale_frequency = st.sidebar.slider('Frecuencia de Venta del Producto', df['ProductSaleFrequency'].min(),
                                 df['ProductSaleFrequency'].max(), int(df['ProductSaleFrequency'].mean()))
    customer_order_count = st.sidebar.slider('Frecuencia de Compra del Cliente', df['CustomerOrderCount'].min(),
                                     df['CustomerOrderCount'].max(), int(df['CustomerOrderCount'].mean()))
    price = st.sidebar.slider('Precio', df['Price'].min(),
                                     df['Price'].max(), df['Price'].mean())
    quantity = st.sidebar.slider('Cantidad', df['Quantity'].min(),
                                     df['Quantity'].max(), int(df['Quantity'].mean()))
    # Predict
    if st.sidebar.button('Predecir'):
        predicted_revenue = predict_revenue(model, month, is_weekend,product_sale_frequency, customer_order_count, country, price, quantity)
        st.sidebar.success(f'Ventas {predicted_revenue:.2f}')
        print(f'Predicted Revenue: {predicted_revenue:.2f}')

if __name__ == '__main__':
    main()
