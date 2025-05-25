from flask import Flask, request, jsonify, send_from_directory
import os
import torch
from werkzeug.utils import secure_filename
from flask_cors import CORS
from main import process_video, classify_videos 
import logging
import threading
import json
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={
    r"/upload": {"origins": ["http://localhost:3000"]},
    r"/status/*": {"origins": ["http://localhost:3000"]},
    r"/video/*": {"origins": ["http://localhost:3000"]},
    r"/frame/*": {"origins": ["http://localhost:3000"]}
})
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'extracted_videos_output'
FRAME_FOLDER = 'final_frames'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['FRAME_FOLDER'] = FRAME_FOLDER


# Configure logging
logging.basicConfig(level=logging.DEBUG)

processing_status = {}

def process_video_async(file_path, output_folder):
    video_name = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        # Update status - processing started
        update_status(video_name, {
            "progress": 10,
            "stage": "initial_processing"
        })
        
        torch.cuda.empty_cache()
        
        # Update status - frame extraction
        update_status(video_name, {
            "progress": 30,
            "stage": "extracting_frames"
        })
        
        frame_paths, bounce_results, ball_video_paths = process_video(file_path, output_folder)
        
        # Update status - classification
        update_status(video_name, {
            "progress": 70,
            "stage": "classifying_shots"
        })
        
        results = classify_videos(video_name, output_folder)
        # prepare each frame file_name
        frame_data = []
        print("frame_paths:", frame_paths)
        for frame_path in frame_paths:
            frame_filename = os.path.basename(frame_path)
            # frame_url = f"../{OUTPUT_FOLDER}/{os.path.splitext(os.path.basename(file_path))[0]}/final_frames/{frame_filename}"
            frame_data.append({"frame": frame_filename})

        # prepare each ball video

        ball_video = []
        for ball_file_path in ball_video_paths:
            ball_file_name = os.path.basename(ball_file_path)
            ball_video.append({f"ball_video": ball_file_name})
        print(ball_video_paths)

        # Update processing status


        # processing_status.update(file_path, {
        #     "status": "completed",
        #     "results": results,
        #     "video_name": video_path,
        #     "frame_ranges": frame_ranges,
        #     "bounce_results": bounce_results,
        #     "highlights_video": f"{video_path}_complete_highlights.mp4",
        #     "frame_data": frame_data
        # })
        update_status(video_name, {
            "status": "completed",
            "progress": 100,
            "results": results,
            "ball_videos": ball_video,
            "frame_data": frame_data,
            "bounce_results": bounce_results,
            "end_time": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Exception in processing: {str(e)}")
        update_status(video_name, {
            "status": "error",
            "error": str(e)
        })


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            video_name = os.path.splitext(filename)[0]
            processing_dir = os.path.join(app.config['OUTPUT_FOLDER'], video_name)
            os.makedirs(processing_dir, exist_ok=True)

            # Initialize status file
            status_file = os.path.join(processing_dir, 'processing_status.json')
            
            initial_status = {
                "status": "processing",
                "file_path": file_path,
                "video_name": video_name,
                "progress": 0,
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
                "stage": "uploaded"
            }

            with open(status_file, 'w') as f:
                json.dump(initial_status, f)

            # Start processing in a separate thread
            thread = threading.Thread(target=process_video_async, args=(file_path, app.config['OUTPUT_FOLDER']))
            thread.start()


            return jsonify({
                "message": "File uploaded and processing started",
                "file_path": file_path,
                "video_name": video_name
            })
        
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/status/<video_name>', methods=['GET'])
def get_status(video_name):
    try:
        status_file = os.path.join(
            app.config['OUTPUT_FOLDER'], 
            video_name, 
            'processing_status.json'
        )
        
        if not os.path.exists(status_file):
            return jsonify({"error": "Status file not found"}), 404
        
        with open(status_file, 'r') as f:
            status_data = json.load(f)
            
        return jsonify(status_data)
        
    except Exception as e:
        logging.error(f"Exception in status check: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def update_status(video_name, updates):
    try:
        status_file = os.path.join(
            app.config['OUTPUT_FOLDER'], 
            video_name, 
            'processing_status.json'
        )
        
        with open(status_file, 'r') as f:
            status_data = json.load(f)
            
        status_data.update(updates)
        status_data['last_update'] = datetime.now().isoformat()
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f)

    except Exception as e:
        logging.error(f"Error updating status: {str(e)}")

@app.route('/frame/<video_name>/<filename>')
def serve_frame(video_name, filename):
    frame_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_name, "final_frames")
    response = send_from_directory(frame_folder, filename)
    logging.debug(f"Serving frame: {filename} with MIME type: {response.mimetype}")
    return response

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(debug=False, port=5001)
