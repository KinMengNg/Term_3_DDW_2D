import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

###############################################################
#Functions
###############################################################
def normalize_z(dfin, columns_means=None, columns_stds=None):
    if columns_means is None:
        columns_means = dfin.mean()
    if columns_stds is None:
        columns_stds = dfin.std()
        
    dfout = (dfin - columns_means) / columns_stds
        
    return dfout, columns_means, columns_stds

#feature_names and target_names are lists
def get_features_targets(df, feature_names, target_names):
    df_feature = df[feature_names].copy()
    df_target = df[target_names].copy()
    
    return df_feature, df_target

#Convert to numpy, and add a column of 1s
def prepare_feature(df_feature):
    #Assume df feature CANNOT be a series because get_features_targets makes sure that they are dataframes
    if isinstance(df_feature, pd.DataFrame):
        np_feature = df_feature.to_numpy()
    else:
        np_feature = df_feature
        
    #Get number of rows
    num_rows = np_feature.shape[0]
    #Create a column of ones
    ones_col = np.ones((num_rows, 1))
    
    #Concatenate them horizontally
    X = np.concatenate((ones_col, np_feature), axis=1)
    
    #Return X
    return X
    

def prepare_target(df_target):
    #Assume df target CANNOT be a series because get_features_targets makes sure that they are dataframes
    if isinstance(df_target, pd.DataFrame):
        np_target = df_target.to_numpy()
    else:
        np_target = df_target
        
    return np_target #this is just y

    
#Normalizes features, prepare the data, then do prediction
def predict_linreg(df_feature, beta, means=None, stds=None):
    print(df_feature)

    #Normalize features
    df_feature_z, _, _ = normalize_z(df_feature, means, stds)
    print(df_feature_z)
    
    #Prepare the feature (convert to numpy and add column of 1)
    X = prepare_feature(df_feature_z) #I FORGOT THE Z FFS
    
    print(X)
    print(beta)

    ypred = calc_linreg(X, beta)
    
    return ypred
    

def calc_linreg(X, beta):
    #Its just ypred = Xb #X matrix multiply b
    return np.matmul(X, beta)
    

def split_data(df_feature, df_target, random_state=None, test_size=0.5):
     
    #Get all indexs
    all_indexs = df_target.index #can take from df_feature too, doesnt matter
    
    #Get the test indexs
    np.random.seed(random_state)
    
    test_indexs = sorted(list(np.random.choice(all_indexs, size = int(len(all_indexs)*test_size), replace = False)))
    
    #Start splitting
    df_feature_test = df_feature.loc[test_indexs, : ]
    df_target_test = df_target.loc[test_indexs, : ]
    
    df_feature_train = df_feature.drop(test_indexs)
    df_target_train = df_target.drop(test_indexs)
    
    
    return df_feature_train, df_feature_test, df_target_train, df_target_test
  
def r2_score(y, ypred):
    residuals = y-ypred
    SSres = np.matmul(residuals.T, residuals)
    
    totals = y - np.mean(y)
    SStot = np.matmul(totals.T, totals)
    
    r2 = 1 - (SSres/SStot)
    return r2[0][0]

def mean_squared_error(target, pred):
    error = target - pred
    mse = np.matmul(error.T, error) / len(pred)
    return mse[0][0]


def compute_cost_linreg(X, y, beta):
    #Get predictions
    ypred = calc_linreg(X, beta)
    
    #Get errors (residuals)
    errors = y - ypred
    
    #Calculate cost (loss)
    m = y.shape[0] #Number of rows / datapoints #Can take x.shape[0], its the same
    J = (np.matmul(errors.T, errors)) / (2*m)
    
    return J[0][0]

def gradient_descent_linreg(X, y, beta, alpha, num_iters):
    J_storage = np.zeros(num_iters)
    
    m = y.shape[0]
    for i in range(num_iters):
        #Get the prediction
        ypred = calc_linreg(X, beta)
        
        #Get the error
        errors = ypred - y #MUST BE YPRED - Y NOT THE OTHER WAY AROUND. 
        
        #Get the derivative
        derivs = np.matmul(X.T, errors) / m
        
        #Calculate the new beta
        beta = beta - (alpha*derivs)
        
        #Calculate the cost
        J = compute_cost_linreg(X, y, beta)
        
        #Add cost to Jstorage
        J_storage[i] = J
        
    return beta, J_storage


def load_model(filename):
    with open(filename, "rb") as f1:
        tup = pkl.load(f1)

    return tup

############################################################################