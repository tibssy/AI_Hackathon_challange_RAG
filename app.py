from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def get_health():
    return jsonify({'message': 'Healthy'})


@app.route('/api/message', methods=['GET'])
def get_message():
    return jsonify({'message': 'Hello from the backend!'})


@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.json
    return jsonify({'received': data})


if __name__ == '__main__':
    app.run(debug=True)