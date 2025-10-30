from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from models.model import chatbot, checkpointer
from langchain_core.messages import HumanMessage

ITERATION = 0
MAX_ITERATION = 3
CONFIG = {'configurable': {'thread_id': '1'}}

app =Flask(__name__)
CORS(app)

@app.route('/', methods = ['GET','POST'])
def chat():
    if request.method == 'GET':
        return render_template('index.html')
    
    data = request.get_json()
    
    json = chatbot.invoke(
        {'user_query': HumanMessage(data['user_query']),
         'iteration': ITERATION,
         'max_iteration': MAX_ITERATION}, checkpointer = checkpointer, config=CONFIG
    )

    return jsonify({'ai_message': json['answer']})


if __name__ == '__main__':
    app.run()