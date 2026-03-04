from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

def getXY(df, invars, outvars):
    print(len(df))
    X = df[invars].values
    Y = df[outvars].values
    #print(X,Y)
    return X, Y

def get_block_test_train(df, block_name, invars, outvars):
    """
    Separates the data into training and test sets based on the provided block name.
    
    Parameters:
    - df: DataFrame containing the data.
    - block_name: Name of the block to be used as the test set.
    - invars: List of input variable names.
    - outvars: List of output variable names.
    
    Returns:
    - Tuple containing training and test dataframes.
    """
    df_train = df[df['Blocks'] != block_name]
    df_test = df[df['Blocks'] == block_name]
    
    return df_train[invars + outvars], df_test[invars + outvars]

# Define the variables
#invars = ['Potential_temperature', 'Absolute_salinity', 'Latitude', 'Longitude', 'Pressure','Hab']
#invars2 = ['Potential_temperature', 'Absolute_salinity', 'Latitude', 'Longitude', 'Pressure', 'Hab', 'Oxygen', 'Silicate',  'Phosphate', 'Nitrate']
#outvars = list(df_partitioned.columns[19:-1])

# Test the get_block_test_train function
#df_train, df_test = get_block_test_train(df_partitioned, 0, invars, outvars)
#df_train.head(), df_test.head()



def train_and_evaluate(model, X_train, Y_train, X_test, Y_test):
    """
    Train the model and evaluate its performance.
    
    Parameters:
    - model: A scikit-learn regressor model.
    - X_train, Y_train: Training data.
    - X_test, Y_test: Test data.
    
    Returns:
    - Y_out: Predicted output on the test data.
    - r2: R^2 score of the model on the test data.
    """
    model.fit(X_train, Y_train)
    Y_out = model.predict(X_test)
    r2 = r2_score(Y_test, Y_out)
    return Y_out, r2

def R2(Y_true, Y_pred):
    """Compute R^2 score."""
    return r2_score(Y_true, Y_pred)

def main_training_loop(df, blocks, invars, outvars):
    r2dict1 = {}
    r2dict2 = {}
    r2dict12 = {}

    Youtl1 = []
    Youtl2 = []
    Ytestl = []

    rdiff1 = []
    rdiff2 = []

    for block in blocks:
        df_train, df_test = get_block_test_train(df, block, invars, outvars)
        X_train, Y_train = getXY(df_train, invars, outvars)
        X_test, Y_test = getXY(df_test, invars, outvars)

        # Model 1: DecisionTreeRegressor
        clf1 = DecisionTreeRegressor(max_depth=11)
        Y_out1, r2_1 = train_and_evaluate(clf1, X_train, Y_train, X_test, Y_test)
        Youtl1.append(Y_out1)
        r2dict1[block] = r2_1

        # Model 2: RandomForestRegressor
        clf2 = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=11, 
                                     max_features=1.0, n_jobs=-1)
        Y_out2, r2_2 = train_and_evaluate(clf2, X_train, Y_train, X_test, Y_test)
        Youtl2.append(Y_out2)
        r2dict2[block] = r2_2

        # Average prediction
        Y_out_avg = (Y_out1 + Y_out2) / 2
        r2_avg = R2(Y_test, Y_out_avg)
        r2dict12[block] = r2_avg

        Ytestl.append(Y_test)

        # Compute and store R2 breakup for each model
        rlist1 = [np.floor(R2(Y_test[:, iy], Y_out1[:, iy])) for iy in range(Y_test.shape[1])]
        rlist2 = [np.floor(R2(Y_test[:, iy], Y_out2[:, iy])) for iy in range(Y_test.shape[1])]
        rdiff1.append(rlist1)
        rdiff2.append(rlist2)

        #print(f'Block ID: {block} | R2 Model 1: {np.floor(r2_1*100)} | R2 Model 2: {np.floor(r2_2*100)} | R2 Avg: {np.floor(r2_avg*100)}')
        print(f'Block ID: {block} | R2 Model 1: {(r2_1*100)} | R2 Model 2: {(r2_2*100)} | R2 Avg: {(r2_avg*100)}')

    return r2dict1, r2dict2, r2dict12, Youtl1, Youtl2, Ytestl, rdiff1, rdiff2

# Just a summary function and not actually running it as it might take time.
def summary():
    invars_tot = ['Potential_temperature', 'Absolute_salinity', 'Latitude', 'Longitude', 'Pressure', 'Hab', 
                  'Oxygen', 'Silicate', 'Phosphate', 'Nitrate']
    outvars = list(df_partitioned.columns[19:-1])
    block_nums = df_partitioned['Blocks'].unique()

    return main_training_loop(df_partitioned, block_nums, invars_tot, outvars)

# Displaying function names and their definitions for clarity
from joblib import dump, load

def main_training_loop_with_checkpointing(df, blocks, invars, outvars, checkpoint_path="./models", depth = 18):
    """
    Extended training loop that includes model checkpointing.
    
    Parameters:
    - df: Dataframe containing the data.
    - blocks: Blocks to process.
    - invars: Input variable names.
    - outvars: Output variable names.
    - checkpoint_path: Path to save the model checkpoints.
    
    Returns:
    - Dictionary of R2 scores and other evaluation metrics.
    """
    r2dict1 = {}
    r2dict2 = {}
    r2dict12 = {}

    Youtl1 = []
    Youtl2 = []
    Ytestl = []

    rdiff1 = []
    rdiff2 = []

    for block in blocks:
        df_train, df_test = get_block_test_train(df, block, invars, outvars)
        X_train, Y_train = getXY(df_train, invars, outvars)
        X_test, Y_test = getXY(df_test, invars, outvars)
        print('trainparams:', block, X_train.shape, Y_train.shape)
        print('testparams:', X_test.shape, Y_test.shape)

        # Model 1: DecisionTreeRegressor
        clf1 = DecisionTreeRegressor(max_depth=depth)
        Y_out1, r2_1 = train_and_evaluate(clf1, X_train, Y_train, X_test, Y_test)
        Youtl1.append(Y_out1)
        r2dict1[block] = r2_1

        # Model 2: RandomForestRegressor
        clf2 = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=depth, 
                                     max_features=1.0, n_jobs=-1)
        Y_out2, r2_2 = train_and_evaluate(clf2, X_train, Y_train, X_test, Y_test)
        Youtl2.append(Y_out2)
        r2dict2[block] = r2_2

        # Average prediction
        Y_out_avg = (Y_out1 + Y_out2) / 2
        r2_avg = R2(Y_test, Y_out_avg)
        r2dict12[block] = r2_avg

        Ytestl.append(Y_test)

        # Compute and store R2 breakup for each model
        rlist1 = [np.floor(R2(Y_test[:, iy], Y_out1[:, iy])) for iy in range(Y_test.shape[1])]
        rlist2 = [np.floor(R2(Y_test[:, iy], Y_out2[:, iy])) for iy in range(Y_test.shape[1])]
        rdiff1.append(rlist1)
        rdiff2.append(rlist2)

        print(f'Block ID: {block} | R2 Model 1: {np.floor(r2_1*100)} | R2 Model 2: {np.floor(r2_2*100)} | R2 Avg: {np.floor(r2_avg*100)}')

        # Save (checkpoint) the models
        dump(clf1, f'{checkpoint_path}/dt_block_b{block}_d{depth}.joblib')
        dump(clf2, f'{checkpoint_path}/rf_block_b{block}_d{depth}.joblib')

    results = {
        "r2dict1": r2dict1,
        "r2dict2": r2dict2,
        "r2dict12": r2dict12,
        "Youtl1": Youtl1,
        "Youtl2": Youtl2,
        "Ytestl": Ytestl,
        "rdiff1": rdiff1,
        "rdiff2": rdiff2
    }
    
    return results

# A summary function and not actually running it as it might take time.
def summary_with_checkpointing():
    invars_tot = ['Potential_temperature', 'Absolute_salinity', 'Latitude', 'Longitude', 'Pressure', 'Hab', 
                  'Oxygen', 'Silicate', 'Phosphate', 'Nitrate']
    outvars = list(df_partitioned.columns[19:-1])
    block_nums = df_partitioned['Blocks'].unique()
    checkpoint_dir = "./model_checkpoints"  # Specify your desired path here

    return main_training_loop_with_checkpointing(df_partitioned, block_nums, invars_tot, outvars, checkpoint_dir)

# Displaying function names and their definitions for clarity

