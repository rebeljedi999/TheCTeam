import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/test', methods=['GET', 'POST'])
def test_api():
    print("here")
    print(request.form)
    print("read")
    return jsonify({'meme': 'Hello there'})


@app.route('/handshake', methods=['POST'])
def handshake_api():
    data = request.form

app.run()