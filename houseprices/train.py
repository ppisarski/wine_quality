import pandas as pd
from category_encoders import OrdinalEncoder
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from shapash.data.data_loader import data_loading
from shapash.explainer.smart_explainer import SmartExplainer


def load_data():
    house_df, _ = data_loading('house_prices')
    return house_df


def main():
    df = load_data()
    x = df[df.columns.difference(['SalePrice'])]
    y = df['SalePrice'].to_frame()

    categorical_features = [col for col in x.columns if x[col].dtype == 'object']
    encoder = OrdinalEncoder(
        cols=categorical_features,
        handle_unknown='ignore',
        return_df=True).fit(x)
    x = encoder.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1)
    model = LGBMRegressor(n_estimators=200).fit(x_train, y_train)
    y_pred = pd.DataFrame(model.predict(x_test), columns=['pred'], index=x_test.index)

    xpl = SmartExplainer()
    xpl.compile(x=x_test,
                model=model,
                preprocessing=encoder,
                y_pred=y_pred)
    predictor = xpl.to_smartpredictor()
    xpl.save('../models/houseprices_xpl.pkl')
    predictor.save('../models/houseprices_predictor.pkl')


if __name__ == "__main__":
    main()
