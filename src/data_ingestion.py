import pandas as pd
import logging
import os
import kagglehub
from sklearn.model_selection import train_test_split

# Making Log directory
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging conf
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url: str) -> pd.DataFrame:
    """Load Data From CSV File"""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data Loaded Successfully')
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading the data: %s",e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data"""
    
    try:
        df.drop(columns=['Unnamed: 0'],inplace=True)
        df = df.iloc[:50000]
        logger.debug('Preporcessing Completed')
        return df
    except KeyError as e:
        logger.error('Missing column in Dataframe: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error while preprocessign the data:%s',e)
        raise    
            
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('Train and Test data saved to:%s',raw_data_path)
    except Exception as e:
        logger.error('Unexpected error while saving the data:%s',e)
        raise

def main():
    try:
        test_size=0.2
        path = kagglehub.dataset_download("nelgiriyewithana/emotions")
        data_path = os.path.join(path,os.listdir(path)[0])
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_df,test_df = train_test_split(final_df,test_size=test_size,random_state=2)
        save_data(train_df,test_df,data_path = './data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process:%s',e)
        raise

if __name__ == '__main__':
    main()
        