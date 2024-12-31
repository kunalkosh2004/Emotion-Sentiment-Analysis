import os
import logging
import pandas as pd
import nltk
import string
import spacy
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
# nltk.download('corpus')


# Making Log directory
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging conf
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    # Transforming to lower case
    text = text.lower()
    # Tokenizing the text
    words = nltk.word_tokenize(text)
    
    ps = PorterStemmer()
    
    stop_words = set(stopwords.words('english'))
    # Removing Non-alphanumeric tokens
    text = [word for word in words if word.isalnum()]
    # Removing stopwords and punctuations
    text = [word for word in text if word not in stop_words and word not in string.punctuation]

    # Stemming tokens
    text = [ps.stem(word) for word in text]
    # Joining the tokes
    return " ".join(text)


def preprocess_df(df, text_column="text"):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        
        df[text_column] = df[text_column].apply(transform_text)
        
        logger.debug('Text Column Transformed')
        return df
    except KeyError as e:
        logging.error('Column Not found:%s',e)
        raise
    except Exception as e:
        logging.error('Error During text transformation:%s',e)
        raise

def main(text_column='text',target_column = 'label'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # fetch the raw data
        train_df = pd.read_csv('./data/raw/train.csv')
        test_df = pd.read_csv('./data/raw/test.csv')
        
        # processing the df
        train_pro_df = preprocess_df(train_df)
        test_pro_df = preprocess_df(test_df)
        
        data_path = os.path.join('./data','interm')
        os.makedirs(data_path, exist_ok=True)
        
        train_pro_df.to_csv(os.path.join(data_path,'train_pro.csv'),index=False)
        test_pro_df.to_csv(os.path.join(data_path,'test_pro.csv'),index=False)
        
        logger.debug('Processed Data saved at:%s',data_path)
    except Exception as e:
        logger.error('Error occur while processign DataFrame:%s',e)
        raise

if __name__ == '__main__':
    main()
        