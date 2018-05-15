'''
Created on 21 Mar 2018

@author: jwong
'''

from flask import Flask, request, jsonify
from flask_cors import CORS
from serve import get_model_api 

app = Flask(__name__)
CORS(app)
model_api = get_model_api()

#Default
@app.route('/')
def index():
    return "Index API"

#Handle HTTP Errors
@app.errorhandler(404)
def url_error(e):
    return """
    Unknown site
    <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    Internal Error: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500
    
#API
@app.route('/api', methods=['POST'])
def api():
    print('start API')
    input_data = request.json
    print(input_data)
    output_data = model_api(input_data[0], input_data[1])
    response = jsonify(output_data)
    print('response made: {}'.format(response))
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)