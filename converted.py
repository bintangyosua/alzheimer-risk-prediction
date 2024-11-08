import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    return (
        LabelEncoder,
        StandardScaler,
        accuracy_score,
        classification_report,
        np,
        pd,
        plt,
        sns,
        train_test_split,
        xgb,
    )


@app.cell
def __(pd):
    df = pd.read_csv('alzheimers_disease_data.csv')
    df.head()
    return (df,)


@app.cell
def __(df):
    df.info()
    return


@app.cell
def __(df):
    df_1 = df.drop('DoctorInCharge', axis=1)
    return (df_1,)


@app.cell
def __(df_1):
    X = df_1.drop('Diagnosis', axis=1)
    y = df_1['Diagnosis']
    return X, y


@app.cell
def __(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    return X_test, X_train, y_test, y_train


@app.cell
def __(StandardScaler, X_test, X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, X_train_scaled, scaler


@app.cell
def __(X_train_scaled, xgb, y_train):
    model = xgb.XGBClassifier()
    model.fit(X_train_scaled, y_train, verbose=True)
    return (model,)


@app.cell
def __(X_test_scaled, model):
    y_pred = model.predict(X_test_scaled)
    return (y_pred,)


@app.cell
def __(accuracy_score, y_pred, y_test):
    accuracy_score(y_test, y_pred)
    return


if __name__ == "__main__":
    app.run()

