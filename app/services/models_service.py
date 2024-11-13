from models import train_model
from utils import views
import numpy as np
import joblib
import os
import pandas as pd
from logs_models.config_logs import setup_logger_predict
#from models.train_model import models_predicts_ensamble
from models.train_model import predicts_gradient_models,predicts_random_forest_models,predicts_xgbregressor_models


logger = setup_logger_predict()


data_train_path = "/home/pablopalacios/code/langchain/app/dataset/data_prome.cvs"

filename_df = pd.read_csv(data_train_path)



def prediction_avg_general(at_bats, hits):
    manual = views.avgs_(at_bats,hits)

    input_data = pd.DataFrame(np.array([[at_bats,hits]]), columns=['AB','H'])

    
    gradient_predict = predicts_gradient_models(input_data)
    ramdon_forest_predict = predicts_random_forest_models(input_data)
    xgbregressor_predict = predicts_xgbregressor_models(input_data)
    
    promedio_ia_avg = (gradient_predict + ramdon_forest_predict + xgbregressor_predict)/3

    dic_predict = {
        "promedio_ia_avg":round(promedio_ia_avg,3),
        "manual_avg":manual
        }
   
 
    new_data = {
        "AB": at_bats,
        "H": hits,
        "BA": manual
    }
    new_df = pd.DataFrame([new_data])

    
    data_actualizada = pd.concat([filename_df, new_df], ignore_index=True)
    data_actualizada.to_csv(data_train_path, index=False)

    logger.info("Info prediction avg models")
    logger.info(f"""
                "promedio_ia_avg":{promedio_ia_avg},
                "manual_avg":{manual}""")
                

    return dic_predict





    
        

    




