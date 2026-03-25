class Databundle:

    def __init__(self, X_train, X_test, y_train, y_test, metadata=None):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.metadata = metadata or {}