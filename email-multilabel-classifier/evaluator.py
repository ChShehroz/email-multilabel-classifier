from sklearn.metrics import accuracy_score


def evaluate(model, bundle):

    pred = model.predict(bundle.X_test)

    for key in pred:

        acc = accuracy_score(bundle.y_test[key], pred[key])

        print(f"Accuracy for {key}:", acc)