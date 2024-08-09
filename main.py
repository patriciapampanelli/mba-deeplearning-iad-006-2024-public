from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample endpoint
@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({'message': 'Hello, World!'})

# Example endpoint with parameters
@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get('name', 'Guest')
    return jsonify({'message': f'Hello, {name}!'})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

#comentario 2105
#comentario 2128