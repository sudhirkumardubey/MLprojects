# Main aim is to read data from data source 
import os
import sys # to use custom exception

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass # use a decorator
class DataIngerstionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path : str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngerstionConfig()

    def initiate_data_ingestion(self):
        """To read data from some source"""
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv') # Here we can read data from mongodb, mysql, api or other source
            logging.info("Read the dataset as dataframe")

            # Create artifact folder as data is stored in artifact folder, os.path.dirname = get directory name with given specific path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # save raw data , by taking raw datapath
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            print("completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


# # sanity check:

# if __name__ == "__main__":
#     # logging.info("Devide by zero")
#     obj = DataIngestion()
#     obj.initiate_data_ingestion()
