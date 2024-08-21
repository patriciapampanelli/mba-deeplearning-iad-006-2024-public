# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, request, render_template
from predict_digit import PredictDigit

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    request_data = request.get_json()
    result = PredictDigit(request_data['image'], request_data['model']).call()
    return result

wsgi = WsgiToAsgi(app)

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    wsgi.run()
