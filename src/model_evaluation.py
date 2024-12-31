import os
import logging
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import pandas as pd
import numpy as np
import pickle
import json
from dvclive import Live
import yaml

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging conf
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_param(file_path:str) ->dict:
    """Load parameters from a YAML file."""
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameter retrieved from:%s',file_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_path:str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data loaded from the path: %s',data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def load_model(model_path:str):
    """Load the trained model from a file."""
    try:
        with open(model_path,'rb') as file:
            model = pickle.load(file)
            logger.debug('Model loaded from %s', model_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def evaluate_model(clf,X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        metrics = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred, average='weighted')
        recall = recall_score(y_test,y_pred, average='weighted')
        # roc_auc = roc_auc_score(y_test,y_pred, average='weighted', multi_class='ovr')
        
        metrics_dict = {
            "Accuracy":metrics,
            "precision":precision,
            "recall":recall
        }
        logger.debug('Model Evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise
    
def save_metrics(metrics: dict, file_path:str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path, 'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug('Evaluation metrics is saved at: %s',file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise
    
def main():
    try:
        test_data = load_data('./data/porcessed/test_tfidf.csv')
        
        X_test = test_data.iloc[:,:-1]
        y_test = test_data.iloc[:,-1]
        
        clf = load_model('./models/model.pkl')
        
        metrics = evaluate_model(clf,X_test,y_test)
        params = load_param('params.yaml')
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test, average='weighted'))
            live.log_metric('recall', recall_score(y_test, y_test, average='weighted'))

            live.log_params(params)
        
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Error while evaluating model:%s',e)
        raise
    
if __name__=='__main__':
    main()