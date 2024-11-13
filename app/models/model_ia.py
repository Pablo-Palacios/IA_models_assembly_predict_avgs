from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from api.objects import Modelo

class GradientBoostingModel(Modelo):
    def __init__(self):
        super().__init__(nombre="Gradient Booting Regressor Model")
        self.modelo = GradientBoostingRegressor(n_estimators=150,learning_rate=0.3)


class RandomForestModel(Modelo):
    def __init__(self, nombre="Random Forest Regressor Model"):
        super().__init__(nombre)
        self.modelo = RandomForestRegressor(n_estimators=150)

class XGBRegressorModel(Modelo):
    def __init__(self, nombre="XGB Regressor Model"):
        super().__init__(nombre)
        self.modelo = XGBRegressor(n_estimators=150,learning_rate=0.3)


 
