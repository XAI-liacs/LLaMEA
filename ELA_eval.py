import numpy as np
import xgboost as xgb
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample
import pandas as pd


def preprocess_data(data):
    has_group = False
    if "group" in data:
        group = data["group"]
        data = data.drop("group", axis=1)
        has_group = True
    
    data = data.dropna()
    data = data[data.columns.drop(list(data.filter(regex='costs_runtime')))]
    #data = data.drop("ela_level.mmce_lda_10", axis=1)
    
    
    
    
    if has_group:
        data["group"] = group
    return(data)



def evaluate_function(f, min_x=-5, max_x=5):
    DIM = 5 #change to appropriate dimensionality
    
    model1 = xgb.XGBClassifier(objective="binary:logistic")
    model1.load_model("LLM_data/dimensions/model_Groups_Basins_scaled_new.json")
    
    model2 = xgb.XGBClassifier(objective="binary:logistic")
    model2.load_model("LLM_data/dimensions/model_Groups_Structure_scaled_new.json")
   
    
    problem = f
    X = create_initial_sample(DIM,n=250*DIM, lower_bound = min_x, upper_bound = max_x)
    y = X.apply(problem, axis = 1)
    
    
    y[y==0] = 0.1**100 #since y=0 breaks log
    if y.max() == y.min():
        for i in range(len(y)):
            y[i] = 0
    else:
        X_scaled=(X-X.min())/(X.max()-X.min())
        y_scaled=(y-y.min())/(y.max()-y.min())
    
    
    
    
    ela_meta_scaled = calculate_ela_meta(X_scaled, y_scaled)
    #ela_level = calculate_ela_level(X, y)
    ela_distr_scaled = calculate_ela_distribution(X_scaled, y_scaled)
    
    nbc_scaled = calculate_nbc(X_scaled, y_scaled)
    
    disp_scaled = calculate_dispersion(X_scaled, y_scaled)
    
    pca_scaled = calculate_pca(X_scaled, y_scaled)
    
    ic_scaled = calculate_information_content(X_scaled, y_scaled)
    
    d =  {"dim": DIM} 
    all_features_scaled = {**ela_meta_scaled, **ela_distr_scaled, **nbc_scaled, **disp_scaled, **pca_scaled, **ic_scaled}
    
    
    all_features_scaled = {k:[v] for k,v in all_features_scaled.items()} 
    all_features_scaled = pd.DataFrame.from_dict(all_features_scaled)
    
    all_features_scaled = preprocess_data(all_features_scaled)
    
    
    result_1 = model1.predict_proba(all_features_scaled)
    result_2 = model2.predict_proba(all_features_scaled)
    return(result_1[0][1] + result_2[0][1])



#Example usage
np.random.seed(1)
def test_func(x):
    return np.sum(x**2)
result = evaluate_function(test_func)