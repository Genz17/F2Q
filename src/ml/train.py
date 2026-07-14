import numpy as np
import xgboost as xgb

if __name__ == "__main__":
    data = np.load('xgboost_training_data.npz')
    X_train = data['X']
    y_train = data['y']

    model = xgb.XGBRegressor(
        n_estimators=500,     
        max_depth=20,           
        learning_rate=0.1,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    model.save_model("light_surrogate.json")
