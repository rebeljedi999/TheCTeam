import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route('/test', methods=['GET', 'POST'])
def test_api():
    print("here")
    print(request.form)
    print("read")
    return jsonify({'meme': 'Hello there'})
app.run()
