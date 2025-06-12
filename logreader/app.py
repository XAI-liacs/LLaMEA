import os
import argparse
from datetime import datetime

import jsonlines
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

parser = argparse.ArgumentParser(description="Conversation log viewer")
parser.add_argument(
    "--logfile",
    "-l",
    default=None,
    help="Path to the conversation log (.jsonl) file. Can also be set via CONVERSATION_LOG",
)
args, _ = parser.parse_known_args()

# Determine path to the jsonlines file
MESSAGES_FILE = (
    args.logfile
    or os.environ.get("CONVERSATION_LOG")
    or os.path.join(os.getcwd(), "conversationlog.jsonl")
)

app = Flask(__name__)
socketio = SocketIO(app)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    messages = []
    if os.path.exists(MESSAGES_FILE):
        with jsonlines.open(MESSAGES_FILE) as reader:
            for obj in reader:
                messages.append(obj)
    emit("load_messages", messages)


@socketio.on("new_message")
def handle_new_message(data):
    message = {
        "role": data["role"],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "content": data["content"],
    }
    with jsonlines.open(MESSAGES_FILE, mode="a") as writer:
        writer.write(message)
    emit("new_message", message, broadcast=True)


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)
