import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Melakukan Klasifikasi Penyakit Alzeimer dengan Algoritma `Xgboost` dan `KNN`

        1. Panky Bintang Pradana Yosua (H1D022077)
        2. Anggota 2
        3. Anggota 3
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Import Libraries""")
    return


@app.cell
def __():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.neighbors import KNeighborsClassifier

    import xgboost as xgb
    return (
        KNeighborsClassifier,
        LabelEncoder,
        StandardScaler,
        accuracy_score,
        alt,
        classification_report,
        np,
        pd,
        plt,
        sns,
        train_test_split,
        xgb,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Data Exploration""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Read the dataset

        Data ini memiliki kolom primary key yaitu `PatientID`, memiliki kolom diskrit seperti `Gender`, `Ethnicity`, `EducationLevel`, `dll`, serta kolom continuous seperti `Age`, `BMI`, dan `AlchoholConsumption`
        """
    )
    return


@app.cell(hide_code=True)
def __(pd):
    df = pd.read_csv('alzheimers_disease_data.csv')
    df = df.drop(['DoctorInCharge', 'PatientID'], axis=1)
    df
    return (df,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Explore Kolom-Kolom

        Tidak ada kolom dengan value `null`. Semua kolom memiliki value.
        """
    )
    return


@app.cell
def __(df):
    df.info()
    return


@app.cell
def __(df):
    df.describe()
    return


@app.cell
def __(df, pd):
    columns = df.columns
    continuous_columns_y = [
        col for col in columns if (pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10)
    ]
    return columns, continuous_columns_y


@app.cell
def __(columns, mo):
    x_scatter = mo.ui.dropdown(columns, value=columns[0])
    y_scatter = mo.ui.dropdown(columns, value=columns[1])
    return x_scatter, y_scatter


@app.cell
def __(mo, x_scatter, y_scatter):
    mo.md(f"""
    #### Variable yang akan diplot pada `scatterplot`

    x: {x_scatter} &nbsp;
    y: {y_scatter}
    """)
    return


@app.cell(hide_code=True)
def __(alt, df, mo, x_scatter, y_scatter):
    brush = alt.selection_point(encodings=["x"])

    scatter_bar_chart = (alt.Chart(df)
                          .mark_point()
                          .encode(
                              x=x_scatter.value,
                              y=y_scatter.value,
                              color='Diagnosis')
                          .add_params(brush)
                         )
    scatter_bar_chart = mo.ui.altair_chart(scatter_bar_chart)
    return brush, scatter_bar_chart


@app.cell(hide_code=True)
def __(mo, scatter_bar_chart):
    mo.vstack([scatter_bar_chart, scatter_bar_chart.value.head()])
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Membagi Dataset 

        Membagi dataset Menjadi `X` dan `y`, serta membaginya juga menjadi dataset `training` dan dataset `testing`
        """
    )
    return


@app.cell
def __(df):
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    return X, y


@app.cell
def __(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Normalisasi Dataset

        Tujuannya yaitu supaya bobot masing-masing kolom memiliki skala yang sama, sehingga bias dapat dihindari.

        Pada **KNN** misalnya, skala sangat penting karena **KNN** menggunakan jarak sebagai perhitungan.
        """
    )
    return


@app.cell
def __(StandardScaler, X_test, X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, X_train_scaled, scaler


@app.cell
def __(X_train_scaled):
    X_train_scaled
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Membuat Model

        Model yang akan dibuat disini adalah **XgBoost** dan **KNN (K Nearest Neighbors)**
        """
    )
    return


@app.cell
def __(accuracy_score, mo):
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    mo.md("Fungssi Model Evaluation")
    return (evaluate_model,)


@app.cell(hide_code=True)
def __(mo):
    neighbors = mo.ui.slider(3, 99, 2)
    mo.md(f"""
        Masukkan jumlah neighbors untuk algoritma KNN: {neighbors}
    """)
    return (neighbors,)


@app.cell(hide_code=True)
def __(mo, neighbors):
    mo.md(f"Jumlah Neighbors yang dimasukkan adalah: {neighbors.value}")
    return


@app.cell
def __(KNeighborsClassifier, neighbors, xgb):
    models = {
        "XGBoost": xgb.XGBClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=neighbors.value),
    }
    return (models,)


@app.cell(hide_code=True)
def __(X_test, X_train, evaluate_model, models, y_test, y_train):
    results = {}

    for model_name, model in models.items():
        accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[model_name] = accuracy

    results
    return accuracy, model, model_name, results


@app.cell
def __(pd, results):
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    return (results_df,)


@app.cell
def __(alt, mo, results_df):
    bar_accuracy = mo.ui.altair_chart(
        alt.Chart(results_df, width=300)
        .mark_bar(fill='#4c78a8')
        .encode(
            x=alt.X('Model:O', axis=alt.Axis(labelAngle=0)),
            y='Accuracy'
        )
    )
    bar_accuracy
    return (bar_accuracy,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
