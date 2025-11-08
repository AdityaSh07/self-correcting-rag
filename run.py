from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from models.model import chatbot, checkpointer
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': '1'}}

app =Flask(__name__)
CORS(app)

@app.route('/', methods = ['GET','POST'])
def chat():
    if request.method == 'GET':
        return render_template('index.html')
    
    data = request.get_json()
    
    json = chatbot.invoke(
        {'question': data['user_query'], 'count':0, 'max_count': 2}, checkpointer = checkpointer, config=CONFIG
    )

    return jsonify({'ai_message': json['generation']})


if __name__ == '__main__':
    app.run()