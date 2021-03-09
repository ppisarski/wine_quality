import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from shapash.explainer.smart_explainer import SmartExplainer


def load_data():
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(dataset_url, sep=';')
    return df


def main():
    df = load_data()
    x = df[['fixed acidity', 'volatile acidity', 'citric acid',
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
    y = df['quality']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(max_depth=6, random_state=42, n_estimators=10)
    model.fit(x_train, y_train)
    y_pred = pd.DataFrame(model.predict(x_test), columns=['pred'], index=x_test.index)

    xpl = SmartExplainer()
    xpl.compile(x=x_test,
                model=model,
                y_pred=y_pred)
    xpl.save('../models/winequality-red_xpl.pkl')


if __name__ == "__main__":
    main()
