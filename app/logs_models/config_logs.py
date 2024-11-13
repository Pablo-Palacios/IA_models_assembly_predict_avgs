import logging
import os

def setup_logger(log_file='logs_models/training_models.log'):
    # Crear la carpeta del log si no existe
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configuración del logger
    logger = logging.getLogger('test_logger_')
    logger.setLevel(logging.INFO)

    # Configuración del handler para escribir en el archivo de log
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Formato del log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)


    return logger


def setup_logger_predict(log_file='logs_models/predicts_models.log'):
    # Crear la carpeta del log si no existe
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configuración del logger
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.INFO)

    # Configuración del handler para escribir en el archivo de log
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Formato del log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger