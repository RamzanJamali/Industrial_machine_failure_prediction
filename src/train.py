from sklearn.ensemble import RandomForestClassifier

def train_optimal_rf(X_train, y_train):
    # Using optimal hyperparameters found in the systematic sweep
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=33,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
