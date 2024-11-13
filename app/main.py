from flask import Flask
from api.routes import bp as api_bp


app = Flask(__name__)

app.register_blueprint(api_bp,url_prefix='/api')


#app.config["DEBUG"] = True

if __name__=="__main__":
    app.run(debug=True)

