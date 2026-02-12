import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data(uploaded_test_file=None):

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    # Load training data
    data = pd.read_csv(
        "data/adult.data",
        header=None,
        names=columns,
        na_values=" ?",
        skipinitialspace=True
    )

    data.dropna(inplace=True)

    label_encoders = {}

    # Encode training data
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    X = data.drop("income", axis=1)
    y = data["income"]

    
    # TEST FILE HANDLING
    if uploaded_test_file is not None:
        test_data = pd.read_csv(
            uploaded_test_file,
            header=None,
            names=columns,
            skiprows=1,
            na_values=" ?",
            skipinitialspace=True
        )

        test_data.dropna(inplace=True)

        # Remove trailing dot in income column
        test_data["income"] = test_data["income"].str.replace(".", "", regex=False)

        # Encode using same encoders
        for col in test_data.columns:
            if test_data[col].dtype == "object":
                test_data[col] = label_encoders[col].transform(test_data[col])

        X_train = X
        y_train = y
        X_test = test_data.drop("income", axis=1)
        y_test = test_data["income"]

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
