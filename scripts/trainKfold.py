from sklearn.model_selection import KFold
from joblib import dump
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np
def getXY(df, invars, outvars):
    print(len(df))
    X = df[invars].values
    Y = df[outvars].values
    #print(X,Y)
    return X, Y

def R2(Y_true, Y_pred):
    """Compute R^2 score."""
    return r2_score(Y_true, Y_pred)

def randomized_kfold_training(df, invars, outvars, n_splits=5, model_depth=32, checkpoint_path="./"):
    """
    Performs randomized 5-fold cross-validation training.

    Parameters:
    - df: DataFrame containing the data.
    - invars: Input variable columns.
    - outvars: Output variable columns.
    - n_splits: Number of splits for k-fold cross-validation.
    - model_depth: Depth of the RandomForest model.
    - checkpoint_path: Directory where to save the trained models.
    
    Returns:
    - List of trained models.
    """
    # Randomize the DataFrame rows
    df_randomized = df.sample(frac=1).reset_index(drop=True)

    # Initialize KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store trained models and their validation scores
    trained_models = []
    val_scores = []

    # Iterate over the KFold splits
    for fold, (train_index, val_index) in enumerate(kf.split(df_randomized)):
        print(f"Training on fold {fold + 1}...")
        df_train, df_val = df_randomized.iloc[train_index], df_randomized.iloc[val_index]
        
        # Extract training and validation data
        X_train, Y_train = df_train[invars].values, df_train[outvars].values
        X_val, Y_val = df_val[invars].values, df_val[outvars].values
        
        print(outvars)

        # Train the RandomForest model
        model = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=model_depth, verbose=True, n_jobs=-1)
        model.fit(X_train, Y_train)

        # Validate the model and store the R^2 score
        Y_val_pred = model.predict(X_val)
        r2_val = R2(Y_val, Y_val_pred)
        val_scores.append(r2_val)

        # Save the trained model
        model_filename = os.path.join(checkpoint_path, f"RF_depth{model_depth}_fold{fold}.joblib")
        dump(model, model_filename)
        trained_models.append(model_filename)

        print(f"Fold {fold + 1} validation R^2: {r2_val}")

    return trained_models, val_scores

from joblib import load

def ensemble_inference(models, X_new):
    """
    Applies ensemble of trained models on new data to compute the mean and variance of predictions.

    Parameters:
    - models: List of trained model filenames.
    - X_new: New input data for predictions.

    Returns:
    - Mean and variance of predictions.
    """
    # List to store predictions from all models
    predictions = []

    # Load each trained model and make predictions on new data
    for model_filename in models:
        model = load(model_filename)
        pred = model.predict(X_new)
        predictions.append(pred)

    # Convert to numpy array for computation
    predictions = np.array(predictions)

    # Compute mean and variance of predictions
    mean_prediction = np.mean(predictions, axis=0)
    variance_prediction = np.var(predictions, axis=0)

    return mean_prediction, variance_prediction

