import pandas as pd
from models.model_ia import RandomForestModel,GradientBoostingModel,XGBRegressorModel
from api.objects import Modelo
from logs_models.config_logs import setup_logger

logger = setup_logger()


data_train_1 = "/home/pablopalacios/code/langchain/app/dataset/data_set_mlb.cvs"
data_train_2 = "/home/pablopalacios/code/langchain/app/dataset/data_prome.cvs"

filename_1 = pd.read_csv(data_train_1)
filename_2 = pd.read_csv(data_train_2)

filename = pd.concat([filename_1, filename_2], ignore_index=True)

a = filename[['AB','H']]
b = filename['BA']

if b.isnull().sum() > 0:
        a = a[b.notna()]
        b = b.dropna()


a_train, a_test, b_train, b_test = Modelo.split_data_train_test(a,b,test_size=0.2,random_state=42)

model_gradient = GradientBoostingModel()
model_random_forest = RandomForestModel()
model_xgbregressor = XGBRegressorModel()

Modelo.pretrain(model_gradient,a_train,b_train)
Modelo.pretrain(model_random_forest,a_train,b_train)
Modelo.pretrain(model_xgbregressor,a_train,b_train)

def predicts_gradient_models(data_set):
        predict = float(Modelo.prediction(model_gradient,data_set))
        return round(predict,3)

def predicts_random_forest_models(data_set):
        predict = float(Modelo.prediction(model_random_forest,data_set))
        return round(predict,3)

def predicts_xgbregressor_models(data_set):
        predict = float(Modelo.prediction(model_xgbregressor,data_set))
        return round(predict,3)

def data_pretrain_models_gradient():
        
        score_gradient = Modelo.score(model_gradient,a_train,b_train)
        mean_squared_gradient = Modelo.mean_squared_error_(model_gradient,b_test,model_gradient.prediction(a_test))
        mean_absolute_gradient = Modelo.mean_absolute_error_(model_gradient,b_test,model_gradient.prediction(a_test))
        r2_score_gradient = Modelo.r2_score_(model_gradient,b_test,model_gradient.prediction(a_test))
        root_mean_squared_gradient = Modelo.root_mean_squared_error_(model_gradient,b_test,model_gradient.prediction(a_test))


        dicts = {
                 "score":score_gradient,
                 "mean_squared_error":mean_squared_gradient,
                 "mean_absolute_error":mean_absolute_gradient,
                 "r2_score":r2_score_gradient,
                 "root_mean_squared_error":root_mean_squared_gradient
        }

        logger.info("Info data pre train models gradient")
        logger.info(f"""
                "score_gradient":{score_gradient},
                "mean_squared_gradient":{mean_squared_gradient},
                "mean_absolute_gradient":{mean_absolute_gradient},
                "r2_score_gradient":{r2_score_gradient},
                "root_mean_squared_gradient":{root_mean_squared_gradient}""")
      
      
        return dicts


def data_pretrain_models_random_forest():
        
        score_random_forest = Modelo.score(model_random_forest,a_train,b_train)
        mean_squared_random_forest = Modelo.mean_squared_error_(model_random_forest,b_test,model_random_forest.prediction(a_test))
        mean_absolute_random_forest = Modelo.mean_absolute_error_(model_random_forest,b_test,model_random_forest.prediction(a_test))
        r2_score_random_forest = Modelo.r2_score_(model_random_forest,b_test,model_random_forest.prediction(a_test))
        root_mean_squared_random_forest = Modelo.root_mean_squared_error_(model_random_forest,b_test,model_random_forest.prediction(a_test))


        dicts = {
                 "score":score_random_forest,
                 "mean_squared_error":mean_squared_random_forest,
                 "mean_absolute_error":mean_absolute_random_forest,
                 "r2_score":r2_score_random_forest,
                 "root_mean_squared_error":root_mean_squared_random_forest

        }

        logger.info("Info data pre train models random forest")
        logger.info(f"""
                "score_random_forest":{score_random_forest},
                "mean_squared_random_forest":{mean_squared_random_forest},
                "mean_absolute_random_forest":{mean_absolute_random_forest},
                "r2_score_random_forest":{r2_score_random_forest},
                "root_mean_squared_random_forest":{root_mean_squared_random_forest}""")
      
        return dicts



def data_pretrain_models_xgbregressor():
        
        
        score_xgbregressor = Modelo.score(model_xgbregressor,a_train,b_train)
        mean_squared_xgbregressor = Modelo.mean_squared_error_(model_xgbregressor,b_test,model_xgbregressor.prediction(a_test))
        mean_absolute_xgbregressor = Modelo.mean_absolute_error_(model_xgbregressor,b_test,model_xgbregressor.prediction(a_test))
        r2_score_xgbregressor = Modelo.r2_score_(model_xgbregressor,b_test,model_xgbregressor.prediction(a_test))
        root_mean_squared_xgbregressor = Modelo.root_mean_squared_error_(model_xgbregressor,b_test,model_xgbregressor.prediction(a_test))


        dicts = {
                 "score":score_xgbregressor,
                 "mean_squared_error":mean_squared_xgbregressor,
                 "mean_absolute_error":mean_absolute_xgbregressor,
                 "r2_score":r2_score_xgbregressor,
                 "root_mean_squared_error":root_mean_squared_xgbregressor
        }

        logger.info("Info data pre train models xgbregressor")
        logger.info(f"""
                "score_xgbregressor":{score_xgbregressor},
                "mean_squared_xgbregressor":{mean_squared_xgbregressor},
                "mean_absolute_xgbregressor":{mean_absolute_xgbregressor},
                "r2_score_xgbregressor":{r2_score_xgbregressor},
                "root_mean_squared_xgbregressor":{root_mean_squared_xgbregressor}""")
      
        return dicts



# joblib.dump(modelo_grandient, "gradient_boosting_model.pkl")
# joblib.dump(modelo_forest, "ramdon_forest_model.pkl")
# joblib.dump(modelo_xgb, "xgbregressor_model.pkl")

        # logger.info("Info data pre train models ia")
        # logger.info(f"""
        #         "mean_squared_grandient":{mean_squared_grandient},
        #         "mean_squared_forest":{mean_squared_forest},
        #         "mean_squared_xgbregressor":{mean_squared_xgb},
        #         "score_grandient":{score_grandient},
        #         "score_forest":{score_forest},
        #         "score_xgbregressor":{score_xgb}""")

# for handler in logger.handlers:
#         handler.flush()

# mean_squared = {
#         "mean_squared_grandient":mean_squared_grandient,
#         "mean_squared_forest":mean_squared_forest,
#         "mean_squared_xgbregressor":mean_squared_xgb,
#         "score_grandient":score_grandient,
#         "score_forest":score_forest,
#         "score_xgbregressor":score_xgb
# }

# with open("datos_train_models.json", "w") as g:
#         json.dump(mean_squared, g)