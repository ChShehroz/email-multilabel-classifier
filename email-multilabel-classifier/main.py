import preprocessing
import feature_extraction
import targets
import Databundle
import model
import evaluator


def main():
    df, loaded_files = preprocessing.load_data()
    X, y2, y3, y4, metadata = preprocessing.preprocess_data(df)

    X_train, X_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = preprocessing.split_data(X, y2, y3, y4)

    X_train_vec, X_test_vec, vectorizer = feature_extraction.vectorize_text(X_train, X_test)

    y_train_targets = targets.build_targets(y2_train, y3_train, y4_train)
    y_test_targets = targets.build_targets(y2_test, y3_test, y4_test)

    bundle = Databundle.Databundle(
        X_train_vec,
        X_test_vec,
        y_train_targets,
        y_test_targets,
        metadata={
            **metadata,
            "loaded_files": loaded_files,
            "vocab_size": len(vectorizer.vocabulary_)
        }
    )

    print("Loaded files:", ", ".join(loaded_files))
    print("Rows after cleaning:", bundle.metadata["row_count"])
    print("Type 1 unique values:", bundle.metadata["type1_unique"])
    print("Removed rare classes:", bundle.metadata["removed_rare_classes"])
    print("Vocabulary size:", bundle.metadata["vocab_size"])

    model_wrapper = model.Chain_Model()
    model_wrapper.train(bundle)
    evaluator.evaluate(model_wrapper, bundle)


if __name__ == "__main__":
    main()