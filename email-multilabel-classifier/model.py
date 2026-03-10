from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class Chain_Model:

    def __init__(self):
        self.models = {
            "t2": RandomForestClassifier(),
            "t23": RandomForestClassifier(),
            "t234": RandomForestClassifier()
        }
        self.encoders = {
            "t2": LabelEncoder(),
            "t23": LabelEncoder(),
            "t234": LabelEncoder()
        }

    def train(self, bundle):
        for key in self.models:
            y = bundle.y_train[key]
            y_encoded = self.encoders[key].fit_transform(y)
            self.models[key].fit(bundle.X_train, y_encoded)
            
    def predict(self, X):
        pred = {}

        for key in self.models:
            y_pred = self.models[key].predict(X)
            y_pred = self.encoders[key].inverse_transform(y_pred)
            pred[key] = y_pred

        return pred