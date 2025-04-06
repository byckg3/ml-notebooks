import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def liver_dataset():

    data = pd.read_csv( "./dataset/Liver_disease_data.csv" )

    int_columns = data.select_dtypes( include = "int" ).columns
    data[ int_columns ] = data[ int_columns ].astype( "float" )

    return data

def split_data( data, target, test_size = 0.2 ):

    train_data, test_data = train_test_split( data, test_size = test_size, random_state = 0 )
    target_column = target

    y_train = train_data[ target ]
    X_train = train_data.drop( columns = [ target ] )

    y_test = test_data[ target ]
    X_test = test_data.drop( columns = [ target ] )

    return X_train, y_train, X_test, y_test


def train_model( estimator, param_grid, X_train, y_train, X_test, y_test, metrics, model_name, best_n = 3 ):

    grid_search = GridSearchCV( estimator = estimator(  random_state = 42  ),
                                param_grid = param_grid,
                                cv = 5,
                                scoring = metrics, 
                                refit = False,
                                n_jobs = -1 )

    grid_search.fit( X_train, y_train )

    n = best_n
    print( f"top {n} results:" )
    results_df = pd.DataFrame( grid_search.cv_results_ )
    results_df = results_df[ [ "params", f"mean_test_{metrics[ 0 ]}", f"rank_test_{metrics[ 0 ]}"  ] ]

    top_results_df = results_df.nsmallest( n, f"rank_test_{metrics[ 0 ]}" )
    top_params = top_results_df[ "params" ]
    print( top_params )

    for params in top_params:
        with mlflow.start_run():

            mlflow.log_input( mlflow.data.from_pandas( X_train ), 
                            context = "train",
                            )
            mlflow.log_params( params )

            model = estimator( **params )
            model.fit( X_train, y_train )

            y_pred = model.predict( X_test )

            accuracy = accuracy_score( y_test, y_pred )
            precision = precision_score( y_test, y_pred )
            recall = recall_score( y_test, y_pred )
            f1 = f1_score( y_test, y_pred )

            print( "accuracy:", accuracy, 
                   "precision:", precision, 
                   "recall:", recall, 
                   "f1:", f1 )
        
            mlflow.log_metrics( { "accuracy": accuracy,
                                  "recall": recall,
                                  "precision": precision,
                                  "f1": f1 
                                } )
            
            model_signature = infer_signature( X_train, model.predict( X_train ) )

            artifact_folder = "sk_models"
            model_info = mlflow.sklearn.log_model( sk_model = model, 
                                                   artifact_path = artifact_folder,
                                                   registered_model_name = model_name,
                                                   input_example = X_train.iloc[ 0:1 ],
                                                   signature = model_signature,
                                                   pip_requirements = "requirements.txt" )
            
            artifact_uri = mlflow.get_artifact_uri( artifact_folder )
            print( "artifact uri:", artifact_uri )

    mlflow.end_run()