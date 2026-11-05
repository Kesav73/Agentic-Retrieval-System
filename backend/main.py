from typing import List
from flask import Flask, request  
from flask_socketio import SocketIO 
from workflow import run_workflow
import base64
import os
from online_data_process import create_retriever_from_pdf
import re

from dotenv import load_dotenv
load_dotenv(".env")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

retriever = None

def generate_response(query):
    """
    Processes a query and emits responses via socket.io.
    This function runs a workflow with the given query and retrieves a response.
    It then extracts file path and answer from the response. If a file path is found,
    it reads the file, encodes it in base64, and emits it in chunks via socket.io.
    Finally, it emits a complete response message. If no file path is found, it emits
    the answer directly.
    Args:
        query (str): The query to be processed.
    Emits:
        response_chunk (dict): A dictionary containing a chunk of the encoded image.
        response_complete (dict): A dictionary containing the final message and image status.
        response (dict): A dictionary containing the final message or error message.
    Raises:
        Exception: If an error occurs during processing.
    """
    try:
        response = run_workflow(query, retriever)
        print("Response:", response)
        
        file_path_match = re.search(r'file_path: (.*?),\n', response)
        answer_match = re.search(r'answer: (.*)', response) # Extracted values 
        file_path = file_path_match.group(1) if file_path_match else None
        answer = answer_match.group(1) if answer_match else response

        if file_path:                        
            with open(file_path, "rb") as f:                
                encoded_image = base64.b64encode(f.read()).decode("utf-8")
                chunk_size = 1024 * 1024  # 1MB
                for i in range(0, len(encoded_image), chunk_size):                    
                    chunk = encoded_image[i:i + chunk_size]                    
                    socketio.emit("response_chunk", {"chunk": chunk})
                response = {
                    "message": answer,
                    # "image_complete": True
                }                
                socketio.emit("response_complete", response)
        else:
            response = response.split("FINAL ANSWER:", 1)[-1].strip()
            socketio.emit("response", {"message": response})
    except Exception as e:
        print(f"Error processing query: {e}")
        socketio.emit("response", {"message": f"Error processing query: {e}"})



# Track uploaded files in memory for simplicity
file_chunks = {}

@socketio.on('file_start')
def handle_file_start(data):
    file_name = data['fileName']
    text = data['text']
    file_chunks[file_name] = {
        'text': text,
        'chunks': [],
        'total_chunks': None
    }
    print(f"Receiving file: {file_name} with initial text: {text}")

@socketio.on('file_chunk')
def handle_file_chunk(data):
    file_name = data['fileName']
    chunk_number = data['chunkNumber']
    total_chunks = data['totalChunks']
    chunk_data = base64.b64decode(data['chunk'])

    if file_name in file_chunks:
        file_chunks[file_name]['chunks'].append((chunk_number, chunk_data))
        if file_chunks[file_name]['total_chunks'] is None:
            file_chunks[file_name]['total_chunks'] = total_chunks

    print(f"Received chunk {chunk_number + 1}/{total_chunks} for file: {file_name}")

@socketio.on('file_complete')
def handle_file_complete(data):
    try:
        global retriever
        file_name = data['fileName']
        if file_name in file_chunks:
            # Sort chunks by chunk number and write to file
            sorted_chunks = sorted(file_chunks[file_name]['chunks'], key=lambda x: x[0])
            file_path = os.path.join("uploads", file_name)
            
            with open(file_path, "wb") as f:
                for _, chunk in sorted_chunks:
                    f.write(chunk)
            
            print(f"File {file_name} assembled and saved.")
            retriever = create_retriever_from_pdf(file_path)
            generate_response(file_chunks[file_name]['text'])
            # Clean up file_chunks dictionary
            del file_chunks[file_name]


        # socketio.emit('file-status', {'status': f'File {file_name} received and saved successfully.'})
    except Exception as e:
        print(f"Error handling file: {e}")
        socketio.emit('file-status', {'status': f'Error handling file: {e}'})


@socketio.on("message")
def handle_query(data):
    message = data
    print("Query:", message)
    generate_response(message)

    socketio.emit("end", {"message": "STOP"})
    # return 'Query received and processing.'


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")    



if __name__ == "__main__":    
    socketio.run(app, host="0.0.0.0", allow_unsafe_werkzeug=True)
