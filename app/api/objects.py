from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

class Modelo():
    def __init__(self,nombre=""):
        self.nombre = nombre
        self.modelo = None
        self.entrenado = False

    
    def pretrain(self,a_train,b_train):
        if self.modelo is None:
            raise NotImplementedError("No se definio el modelo")
        self.modelo.fit(a_train,b_train)
        self.entrenado = True

    def prediction(self,data_test):
        if self.entrenado == False:
            raise NotImplementedError("El modelo no fue entrenado")
        self.entrenado = True
        return self.modelo.predict(data_test)
        
        
    def mean_squared_error_(self,b_test,a_predict):
        return mean_squared_error(b_test,a_predict)
    

    def mean_absolute_error_(self,b_test,a_predict):
        return mean_absolute_error(b_test,a_predict)
    
    
    def root_mean_squared_error_(self,b_test,a_predict):
        return root_mean_squared_error(b_test,a_predict)
    
    def r2_score_(self,b_test,a_predict):
        return r2_score(b_test,a_predict)
    
    def score(self,a_train,b_train):
        return self.modelo.score(a_train,b_train)
    
    def split_data_train_test(a,b,test_size="", random_state=""):
        return train_test_split(a,b,test_size=float(test_size),random_state=int(random_state))
    

    
    

