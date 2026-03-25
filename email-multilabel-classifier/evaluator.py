from sklearn.metrics import accuracy_score, classification_report


def evaluate(model_wrapper, bundle):
    predictions = model_wrapper.predict(bundle.X_test)

    print("\n=== Evaluation Results ===")
    for key in ["t2", "t23", "t234"]:
        score = accuracy_score(bundle.y_test[key], predictions[key])
        print(f"{key} accuracy: {score:.4f}")

    print("\n=== Detailed Classification Report For t2 ===")
    print(classification_report(bundle.y_test["t2"], predictions["t2"], zero_division=0))