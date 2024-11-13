from flask import Blueprint,request,jsonify
from services import models_service
import numpy as np
import pandas as pd
from logs_models.config_logs import setup_logger_predict
from models.train_model import predicts_gradient_models,predicts_random_forest_models,predicts_xgbregressor_models
from models.train_model import data_pretrain_models_gradient,data_pretrain_models_random_forest,data_pretrain_models_xgbregressor

bp = Blueprint('api',__name__)


logger = setup_logger_predict()

@bp.route("/predict_avg", methods=["POST"])
def predict_avg():
    data = request.get_json()
    at_bats = data.get('at_bats')
    hits = data.get('hits')

    if at_bats is None or hits is None:
        return jsonify({"error": "No se pueden dejar vacios los campos"}), 400

    predicts = models_service.prediction_avg_general(at_bats,hits)
    return jsonify({
        "data":predicts
    }), 200


@bp.route("/data_train_models", methods=["POST"])
def get_data_train():
    data = request.get_json()
    model = data.get('model')
    
    if model is None:
        return jsonify({"error": "Se debe especificar modelo: 1- Gradient, 2- Forest, 3- XGBRegressor"}), 400
    
    model = model.lower()

    if model == "gradient":
        gradient_data = data_pretrain_models_gradient()
        dic = {"gradient_data":gradient_data}
        return jsonify({
            "data":dic
        }), 200
    elif model == "forest":
        ramdon_forest_data = data_pretrain_models_random_forest()
        dic = {"ramdon_forest_data":ramdon_forest_data}
        return jsonify({
            "data":dic
        }), 200
    elif model == "xgbregressor":
        xgbregressor_data = data_pretrain_models_xgbregressor()
        dic = {"xgbregressor_data":xgbregressor_data}
        return jsonify({
            "data":dic
        }), 200
    else:
        return jsonify({"error": "Debe elegir entre los modelos: 1- Gradient, 2- Forest, 3- XGBRegressor"}), 404


@bp.route("/predict_avg_model", methods=["POST"])
def predict_avg_model():
    data = request.get_json()
    model = data.get('model')
    at_bats = data.get('at_bats')
    hits = data.get('hits')

    if model is None:
        return jsonify({"error": "Se debe especificar modelo: 1- Gradient, 2- Forest, 3- XGBRegressor"}), 400

    if at_bats is None or hits is None:
        return jsonify({"error": "No se pueden dejar vacios los campos"}), 400
    
    model = model.lower()

    input_data = pd.DataFrame(np.array([[at_bats,hits]]), columns=['AB','H'])

    if model == "gradient":
        gradient_predict = predicts_gradient_models(input_data)
        dic = {"gradient_predict_avg":gradient_predict}

        logger.info("Info prediction avg model gradient")
        logger.info(f"""
                    "gradient_predict":{gradient_predict}
                    """)
        return jsonify({
            "data":dic
        }), 200
    
        
        
    elif model == "forest":
        ramdon_forest_predict = predicts_random_forest_models(input_data)
        dic = {"ramdon_forest_predict":ramdon_forest_predict}

        logger.info("Info prediction avg model ramdon forest")
        logger.info(f"""
                    "ramdon_forest_predict":{ramdon_forest_predict}
                    """)
        
        return jsonify({
            "data":dic
        }), 200
    elif model == "xgbregressor":
        xgbregressor_predict = predicts_xgbregressor_models(input_data)
        dic = {"xgbregressor_predict":xgbregressor_predict}

        logger.info("Info prediction avg model ramdon forest")
        logger.info(f"""
                    "xgbregressor_predict":{xgbregressor_predict}
                    """)
        
        return jsonify({
            "data":dic
        }), 200
    else:
        return jsonify({"error": "Debe elegir entre los modelos: 1- Gradient, 2- Forest, 3- XGBRegressor"}), 404


    