from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import config


class Chain_Model:
    def __init__(self):
        self.models = {
            "t2": RandomForestClassifier(
                n_estimators=config.N_ESTIMATORS,
                random_state=config.MODEL_RANDOM_STATE
            ),
            "t23": RandomForestClassifier(
                n_estimators=config.N_ESTIMATORS,
                random_state=config.MODEL_RANDOM_STATE
            ),
            "t234": RandomForestClassifier(
                n_estimators=config.N_ESTIMATORS,
                random_state=config.MODEL_RANDOM_STATE
            )
        }
        self.encoders = {
            key: LabelEncoder() for key in self.models
        }

    def train(self, bundle):
        for key in self.models:
            y_encoded = self.encoders[key].fit_transform(bundle.y_train[key])
            self.models[key].fit(bundle.X_train, y_encoded)

    def predict(self, X):
        predictions = {}
        for key in self.models:
            y_pred = self.models[key].predict(X)
            predictions[key] = self.encoders[key].inverse_transform(y_pred)
        return predictions