import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import yaml

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging conf
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'feature_engineering.log')
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

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        df.fillna("",inplace=True)
        logger.debug('Data Loaded successfully from:%s',data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame, max_features: int) -> tuple:
    """Apply TfIdf to the data."""
    try:
        vec = TfidfVectorizer(max_features=max_features)
        
        X_train = train_df['text'].values
        y_train = train_df['label'].values
        X_test = test_df['text'].values
        y_test = test_df['label'].values
        
        X_train_bow = vec.fit_transform(X_train)
        X_test_bow = vec.transform(X_test)
        
        train_data = pd.DataFrame(X_train_bow.toarray())
        train_data['label']=y_train
        
        test_data = pd.DataFrame(X_test_bow.toarray())
        test_data['label'] = y_test
        
        logger.debug('Tfidf applied and data transformed')
        return train_data,test_data
    except Exception as e:
        logger.error('Error occured while applying tfidf as :%s',e)
        raise

def save_data(df: pd.DataFrame, file_path:str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise
    
def main():
    try:
        params = load_param('params.yaml')
        max_features = params['feature_engineering']['max_features']
        
        train_data = load_data('./data/interm/train_pro.csv')
        test_data = load_data('./data/interm/test_pro.csv')
        
        train_df,test_df = apply_tfidf(train_data,test_data,max_features)
        
        save_data(train_df,os.path.join('./data','porcessed','train_tfidf.csv'))
        save_data(test_df,os.path.join('./data','porcessed','test_tfidf.csv'))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()