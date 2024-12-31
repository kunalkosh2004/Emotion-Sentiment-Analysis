import os
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging conf
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data is loaded from:%s',data_path)
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
    
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if(X_train.shape[0]!=y_train.shape[0]):
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug('Initailizing RandomForestClassifier')
        clf = RandomForestClassifier(n_estimators=80,random_state=22)
        clf.fit(X_train,y_train)
        
        logger.debug('Model training completed')
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model,file_path:str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as f:
            pickle.dump(model,f)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        train_data = load_data('./data/porcessed/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1]
        y_train = train_data.iloc[:,-1]
        
        clf = train_model(X_train,y_train)
        
        model_save_path = 'models/model.pkl'
        save_model(clf,model_save_path)
        
        logger.debug('Model Build and saved to: %s',model_save_path)
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")
        
if __name__=='__main__':
    main()