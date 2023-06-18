import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# to make pipeline
from sklearn.compose import ColumnTransformer # To create Pipeline

# to handle missing data
from sklearn.impute import SimpleImputer # different imputation methods can be tried 

# to implement pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


"""
    > OneHotEncoder is used for converting categorical values into numerical values.

    -> StandardScalar is used for scaling our data so that our data will be in 
    a specific range and easy for our model to generalize and learn.

    -> @dataclass is basically used for creating class variables.
"""

@dataclass
class DataTransformationConfig:
    """ This will give any path that will be required for data transformation """
    preprocessor_obj_file_path :str = os.path.join('artifacts', "preprocessor.pkl")

class DataTranformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_tranformer_obj(self):
        """
        TO create all my pickle files
        The function is responsible for data transformation
        
        """
        try:
            # numerical feature 
            numerical_columns = ["writing_score", "reading_score"]

            # catergorical feature
            categorical_columns =[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]     

            # create a pipepline
            # To handle missing values, normalise the data, etc
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), # to handle missing values
                    ("scaler", StandardScaler()) # normalization
                ]
            ) 
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scalar", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns standard scaling completed: {numerical_columns}")

            logging.info(f"Categorical columns encoding completed :{categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns), # pipeline name, which pipeline using i.e num_pipeline, and give data to it numerical_Columns
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Pipeline is ready")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_tranformer_obj()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns=[target_column_name], axis =1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # np.c_ concatenate tranformed train and test data

            logging.info("Saved preprocessing object")

            # save_object : funct will write in utils
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj # since it has our all preprocessing inside it
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        