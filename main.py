import cv2
import subprocess
# import pytesseract
import random
import os
from collections import defaultdict
from ultralytics import YOLO
from transformers import VideoMAEForVideoClassification, VivitImageProcessor, VivitForVideoClassification
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
# from deep_sort_realtime.deepsort_tracker import DeepSort


# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load pitch and score_bar multi detection YOLOv8 model
model = YOLO("Yolov8_model.pt")
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ball_model = YOLO('Crciket_ball_tracking/Cricket-ball-tracking-updated-main/runs/detect/train2/weights/best.pt')
ball_model.to(device)


local_model_dir_MAE = "./videomae-base-finetuned-Custom_Dataset_Finetune"
# local_model_vivit = "./Cricket_Shot_Detection_vivit_finetuned_1"
local_model_vivit = "./vivit-b-16x2-kinetics400-Finetune_10Shots"

# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Update this path as needed

# 2. Load the processor
processor_load = VivitImageProcessor.from_pretrained(local_model_vivit)


# Load the model from the local directory
model_load_VideoMAE = VideoMAEForVideoClassification.from_pretrained(local_model_dir_MAE).to(device)
model_load_vivit = VivitForVideoClassification.from_pretrained(local_model_vivit).to(device)


model.confidence = 70  # Increased confidence threshold
model.overlap = 10      # Decreased overlap threshold
BALL_CONF_THRESHOLD = 0.1 # Ball detection confidence threshold



def run_inference(model, frames):
    """Utility to run inference given a model and test video.

    The video is assumed to be preprocessed already.
    """
    # frames = video.unsqueeze(0)

    # 4. Prepare input for the model
    # inputs = processor(frames, return_tensors="pt", sampling_rate=25)
    model.eval()
    # inputs = {
    #     "pixel_values": video.unsqueeze(0),
    #     # "labels": torch.tensor([sample_test_video["label"]]),  # this can be skipped if you don't have labels available.
    # }
    device = torch.device("cpu")
    model = model.to(device)
    # forward pass
    with torch.no_grad():
        outputs = model(pixel_values=frames)
        logits = outputs.logits

    return logits




def get_predicted_class(model, logits):
    """Utility to get the predicted class from logits."""
    # Get the class with the highest logit value
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    # predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    # # Get the class label from the model's configuration
    # class_label = model.config.id2label[predicted_class_idx]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class_idx].item()
    class_label = model.config.id2label[predicted_class_idx]
    return class_label, predicted_class_idx, confidence



def convert_frames_to_videos(frames_directory, video_name, video_output_directory):
    frames = [f for f in os.listdir(frames_directory) if f.endswith('.jpg')]
    frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    sequence_start = 0
    sequence_length = 0
    video_count = 0
    frame_ranges = []
    max_gap = 20
    video_paths = []  # List to store all video paths

    for i in range(len(frames)):
        if i == 0 or (int(frames[i].split('_')[-1].split('.')[0]) - int(frames[i-1].split('_')[-1].split('.')[0]) <= max_gap + 1):
            sequence_length += 1
        else:
            if sequence_length > 20:
                start_frame = int(frames[sequence_start].split('_')[-1].split('.')[0])
                end_frame = int(frames[i-1].split('_')[-1].split('.')[0])
                frame_ranges.append((start_frame, end_frame))
                video_path = create_video_from_sequence(frames_directory, frames[sequence_start:i], video_name, video_count, video_output_directory)
                video_paths.append(video_path)  # Add the video path to the list
                video_count += 1
            sequence_start = i
            sequence_length = 1

    if sequence_length > 20:
        start_frame = int(frames[sequence_start].split('_')[-1].split('.')[0])
        end_frame = int(frames[-1].split('_')[-1].split('.')[0])
        frame_ranges.append((start_frame, end_frame))
        video_path = create_video_from_sequence(frames_directory, frames[sequence_start:], video_name, video_count, video_output_directory)
        video_paths.append(video_path)  # Add the video path to the list

    save_frame_ranges(video_name, frame_ranges, video_output_directory)

    return frame_ranges, video_count, video_paths

def create_video_from_sequence(frames_directory, frames, video_name, video_count, video_output_directory):

    # public_dir = "../public/videos"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec to mp4v for MP4 format
    frame = cv2.imread(os.path.join(frames_directory, frames[0]))
    if frame is None:
        logging.error(f"Error reading frame: {os.path.join(frames_directory, frames[0])}")
        raise ValueError(f"Error reading frame: {os.path.join(frames_directory, frames[0])}")
    height, width, layers = frame.shape
    video_output_path = os.path.join(video_output_directory, f"{video_name}_output_{video_count}.mp4")
    # public_dir = os.path.join(video_output_directory, f"{video_name}_output_{video_count}.mp4")
    out = cv2.VideoWriter(video_output_path, fourcc, 12.0, (width, height))
    # out = cv2.VideoWriter(public_dir, fourcc, 12.0, (width, height))


    for frame_file in frames:
        frame = cv2.imread(os.path.join(frames_directory, frame_file))
        if frame is None:
            logging.error(f"Error reading frame: {os.path.join(frames_directory, frame_file)}")
            raise ValueError(f"Error reading frame: {os.path.join(frames_directory, frame_file)}")
        out.write(frame)

    out.release()
    logging.debug(f"Video {video_count} has been created and saved as {video_output_path}")
    return video_output_path

def save_frame_ranges(video_name, frame_ranges, video_output_directory):
    # Save frame range information to a text file
    frame_ranges_file = os.path.join(video_output_directory, f"{video_name}_frame_ranges.txt")
    with open(frame_ranges_file, 'w') as f:
        for i, (start_frame, end_frame) in enumerate(frame_ranges):
            f.write(f"Video {i}: Start Frame = {start_frame}, End Frame = {end_frame}\n")
    print(f"Frame range information has been saved to {frame_ranges_file}")

def uniform_sampling(frames, num_frames=16):
    if not frames:
        raise ValueError("The frames list is empty. No frames were extracted from the video.")

    total_frames = len(frames)
    if total_frames > num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = [frames[i] for i in frame_indices]
    elif total_frames < num_frames:
        padding = num_frames - total_frames
        last_frame = frames[-1]
        frames.extend([last_frame] * padding)
    return frames


def pitch_coordinates(results):
    """
    Return the bounding box (x1, y1, x2, y2) of the 'Cricket_pitch' detection
    that has the highest confidence.
    """
    pitch_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_label = model.names[cls_id]
            conf = float(box.conf[0])
            if cls_label == "Cricket_pitch":
                x_center, y_center, w, h = box.xywh[0].tolist()
                pitch_boxes.append((conf, x_center, y_center, w, h))
    if not pitch_boxes:
        print("Error: No pitch detected.")
        return None, None, None, None
    pitch_boxes.sort(key=lambda x: x[0], reverse=True)
    best_box = pitch_boxes[0]
    conf, x_center, y_center, w, h = best_box
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = x1 + int(w)
    y2 = y1 + int(h)
    print(f"Selected pitch box with confidence {conf:.2f}: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
    return x1, y1, x2, y2




def detect_ball(ball_model, frame):
    # Perform ball detection
    result = ball_model(frame)
    return result



def draw_pitch_length_annotations(frame, pitch_x1, pitch_y1, pitch_x2, pitch_y2):

    colors = {
        "Short": (0, 0, 255),
        "Good": (0, 255, 0),
        "Full": (255, 0, 0),
        "Yorker": (0, 255, 255)
    }
    h, w = frame.shape[:2]
    x1 = max(0, min(pitch_x1, w))
    y1 = max(0, min(pitch_y1, h))
    x2 = max(0, min(pitch_x2, w))
    y2 = max(0, min(pitch_y2, h))
   # Adjusted pitch length zones based on new percentages
    yorker_length_y = int(y1 + 0.10 * (y2 - y1))
    full_length_y   = int(y1 + 0.15 * (y2 - y1))
    good_length_y   = int(y1 + 0.25 * (y2 - y1))
    short_length_y  = int(y1 + 0.40 * (y2 - y1))
    short_length_y2 = int(y2-100)
    cv2.rectangle(frame, (x1, short_length_y), (x2, short_length_y2), colors["Short"], 2)
    cv2.rectangle(frame, (x1, good_length_y), (x2, short_length_y), colors["Good"], 2)
    cv2.rectangle(frame, (x1, full_length_y), (x2, good_length_y), colors["Full"], 2)
    cv2.rectangle(frame, (x1, yorker_length_y), (x2, full_length_y), colors["Yorker"], 2)
    cv2.putText(frame, "Short", (x1 + 10, short_length_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Short"], 2)
    cv2.putText(frame, "Good", (x1 + 10, good_length_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Good"], 2)
    cv2.putText(frame, "Full", (x1 + 10, full_length_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Full"], 2)
    cv2.putText(frame, "Yorker", (x1 + 10, yorker_length_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Yorker"], 2)


    

def recalc_hit_pitch_annotation_custom(hit_pitch):

    # Define percentages.
    yorker_percentage = 10   # top 10%
    full_percentage = 15    # next 25%
    length_percentage = 10 # next 10%
    short_percentage =40 # bottom 40%
    
    hit_x1, hit_y1, hit_x2, hit_y2 = hit_pitch
    H_hit = hit_y2 - hit_y1

    # Calculate pixel heights for each zone.
    yorker_height = (yorker_percentage / 100) * H_hit
    full_height = (full_percentage / 100) * H_hit
    length_height = (length_percentage / 100) * H_hit
    short_height = H_hit - (yorker_height + full_height + length_height)

    # Determine zone boundaries.
    yorker_start = (hit_y1 + 20)
    yorker_end = yorker_start + yorker_height

    full_start = yorker_end
    full_end = full_start + full_height

    length_start = full_end
    length_end = length_start + length_height

    short_start = length_end
    short_end = hit_y2

    updated_boundaries = {
        "x1": hit_x1,
        "y1": hit_y1,
        "x2": hit_x2,
        "y2": hit_y2,
        "Yorker": (int(yorker_start), int(yorker_end)),
        "Full": (int(full_start), int(full_end)),
        "Length": (int(length_start), int(length_end)),
        "Short": (int(short_start), int(short_end))
    }
    return updated_boundaries

def draw_updated_pitch_annotations(frame, updated_boundaries):

    colors = {
        "Short": (0, 0, 255),
        "Length": (0, 255, 0),
        "Full": (255, 0, 0),
        "Yorker": (0, 255, 255)
    }
    x1 = updated_boundaries["x1"]
    y1 = updated_boundaries["y1"]
    x2 = updated_boundaries["x2"]
    y2 = updated_boundaries["y2"]
    short_top, short_bottom = updated_boundaries["Short"]
    length_top, length_bottom = updated_boundaries["Length"]
    full_top, full_bottom = updated_boundaries["Full"]
    yorker_top, yorker_bottom = updated_boundaries["Yorker"]

    cv2.rectangle(frame, (x1, short_top), (x2, short_bottom), colors["Short"], 2)
    cv2.rectangle(frame, (x1, length_top), (x2, length_bottom), colors["Length"], 2)
    cv2.rectangle(frame, (x1, full_top), (x2, full_bottom), colors["Full"], 2)
    cv2.rectangle(frame, (x1, yorker_top), (x2, yorker_bottom), colors["Yorker"], 2)
    cv2.putText(frame, "Short", (x1 + 10, short_bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Short"], 2)
    cv2.putText(frame, "Length", (x1 + 10, length_bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Length"], 2)
    cv2.putText(frame, "Full", (x1 + 10, full_bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Full"], 2)
    cv2.putText(frame, "Yorker", (x1 + 10, yorker_bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Yorker"], 2)


def classify_bounce_by_position(ball_y, boundaries):
 
    if boundaries["Yorker"][0] <= ball_y <= boundaries["Yorker"][1]:
        return "Yorker Bounce"
    elif boundaries["Full"][0] < ball_y <= boundaries["Full"][1]:
        return "Full Bounce"
    elif boundaries["Length"][0] < ball_y <= boundaries["Length"][1]:
        return "Length Bounce"
    elif boundaries["Short"][0] < ball_y <= boundaries["Short"][1]:
        return "Short Bounce"
    else:
        return "Couldn't bounce classify"





def process_ball_sequence_transition(ball_frames, ball_number,video_name, output_directory, pitch_coords_fixed):

    # Save first frame (reference) with fixed pitch zones.
    frame_paths = []  # List to store paths of saved frames
    original_frames_dir = os.path.join(output_directory, "original_frames")
    annotated_frames_dir = os.path.join(f"../public/{video_name}", "annotated_frames")
    bounce_results_dir = os.path.join(f"../public/", video_name)
    
    os.makedirs(original_frames_dir, exist_ok=True)
    os.makedirs(annotated_frames_dir, exist_ok=True)
    os.makedirs(bounce_results_dir,exist_ok=True)

    first_frame = ball_frames[0].copy()
    draw_pitch_length_annotations(first_frame, *pitch_coords_fixed)
    annotation_file = os.path.join(original_frames_dir, f"ball_{ball_number}_pitch_annotation.jpg")
    cv2.imwrite(annotation_file, first_frame)
    print(f"Ball {ball_number}: Fixed pitch annotation saved at {annotation_file}")
    
    candidate_index = None
    ball_y_values = {}
    prev_y = None
    trend_increasing = False
    hit_candidates = []

    # Process each frame in the ball sequence.
    for frame_idx, frame in enumerate(ball_frames):
        ball_result = detect_ball(ball_model, frame)
        if ball_result and ball_result[0].boxes:
            valid_boxes = [box for box in ball_result[0].boxes if box.conf >= BALL_CONF_THRESHOLD]
            if valid_boxes:
                sorted_boxes = sorted(valid_boxes, key=lambda x: x.conf, reverse=True)
                best_box = sorted_boxes[0]
                _, y1, _, y2 = map(int, best_box.xyxy[0])
                current_y = (y1 + y2) / 2
                ball_y_values[frame_idx] = current_y

                # Annotate frame with y_ball value.
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, f"y_ball: {current_y:.2f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                ball_frames[frame_idx] = annotated_frame
                print(f"Ball {ball_number} - Frame {frame_idx}: y_ball = {current_y}")

                if prev_y is not None:
                    if current_y > prev_y:
                        trend_increasing = True
                    if trend_increasing and current_y < prev_y:
                        candidate_index = frame_idx - 1
                        break
                prev_y = current_y
        else:
            prev_y = 0
        # Save unannotated frame
        frame_output_path = os.path.join(original_frames_dir, f"ball_{ball_number}_frame_{frame_idx}.jpg")
        cv2.imwrite(frame_output_path, frame)

    
    # *** Candidate index check is moved OUTSIDE the for-loop ***
    if candidate_index is None:
        print(f"Ball {ball_number}: No significant transition detected; no hit frame selected.")
        candidate_index = len(ball_frames) - 1 if len(ball_frames) > 1 else 0
        print(f"Ball {ball_number}: Using frame {candidate_index} as fallback for classification.")
    else:
        print(f"Ball {ball_number}: Selected hit frame {candidate_index} with highest deflection.")

    # Use candidate_index to get the hit frame.
    if candidate_index == 0:
        hit_frame = ball_frames[0].copy()
    else:
        hit_frame = ball_frames[candidate_index].copy()

    # Process the hit frame.
    hit_results = model(hit_frame)
    hit_pitch = pitch_coordinates(hit_results)
    if hit_pitch and None not in hit_pitch:
        print(f"Ball {ball_number}: Hit pitch detected: {hit_pitch}. Recalculating boundaries and drawing updated pitch annotations.")
        updated_boundaries = recalc_hit_pitch_annotation_custom(hit_pitch)
        draw_updated_pitch_annotations(hit_frame, updated_boundaries)
        chosen_pitch_y1 = hit_pitch[1]
        chosen_pitch_y2 = hit_pitch[3]
    else:
        print(f"Ball {ball_number}: Hit pitch detection failed; updated pitch annotations will not be drawn.")
        chosen_pitch_y1 = None
        chosen_pitch_y2 = None

    # Crop hit frame to the pitch region.
    x1, y1, x2, y2 = hit_pitch
    cropped_hit_frame = hit_frame[y1:y2, x1:x2]
    # Redetect ball in the hit frame for bounce classification.
    # hit_ball_result = detect_ball(ball_model, hit_frame)
    frame_output_path = os.path.join(original_frames_dir, f"hit_ball__result_ball_{ball_number}_frame_{frame_idx}.jpg")
    cv2.imwrite(frame_output_path, cropped_hit_frame)

    # if hit_ball_result and hit_ball_result[0].boxes:
    #     valid_boxes = [box for box in hit_ball_result[0].boxes]
    #     if valid_boxes:
    #         sorted_boxes = sorted(valid_boxes, key=lambda x: x.conf, reverse=True)
    #         best_box = sorted_boxes[0]
    #         _, box_y1, _, box_y2 = map(int, best_box.xyxy[0])
    #         hit_ball_y = (box_y1 + box_y2) / 2
    #         classification = classify_bounce_by_position(hit_ball_y, updated_boundaries)
    #         cv2.putText(hit_frame, f"{classification}", (10, 30),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    #         hit_file = os.path.join(output_directory, f"ball_{ball_number}_hit.jpg")
    #         cv2.imwrite(hit_file, hit_frame)
    #         print(f"Ball {ball_number}: Hit frame (frame {candidate_index}) classified as {classification}")
    #         hit_classification = (classification)
    #     else:
    #         hit_classification = None
    # else:
    #     hit_classification = None

    if prev_y and current_y:
        classification = classify_bounce_by_position(prev_y, updated_boundaries)
        cv2.putText(hit_frame, f"{classification}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        hit_file = os.path.join(bounce_results_dir, f"ball_{ball_number}_hit.jpg")
        cv2.imwrite(hit_file, hit_frame)
        print(f"Ball {ball_number}: Hit frame (frame {candidate_index}) classified as {classification}")
        hit_classification = (classification)
    else:
        hit_classification = None

    # Replace candidate frame with annotated hit frame.
    ball_frames[candidate_index] = hit_frame

    # Save annotated frames separately
    for frame_idx, frame in enumerate(ball_frames):
        annotated_frame = frame.copy()
        if frame_idx in ball_y_values:
            cv2.putText(annotated_frame, f"y_ball: {ball_y_values[frame_idx]:.2f}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        annotated_frame_output_path = os.path.join(annotated_frames_dir, f"ball_{ball_number}_frame_{frame_idx}.jpg")
        annotated_frame = cv2.imwrite(annotated_frame_output_path, annotated_frame)
        frame_paths.append(annotated_frame_output_path)
        
    

    # Save y_ball values to a text file
    y_values_file = os.path.join(output_directory, f"ball_{ball_number}_y_values.txt")
    with open(y_values_file, 'w') as f:
        for idx, y_val in ball_y_values.items():
            f.write(f"Frame {idx}: y_ball = {y_val}\n")
    print(f"Ball {ball_number}: y_ball values saved in {y_values_file}")
    print ("Done ========>")

    return hit_classification , frame_paths 



def detect_highlights(frames, ball_model, output_directory, ball_number):

    # Create a subfolder for the ball's highlights
    highlights_dir = os.path.join(output_directory, "highlights", f"ball_{ball_number}")
    os.makedirs(highlights_dir, exist_ok=True)
    
    for idx, frame in enumerate(frames):
        frame_copy = frame.copy()
        ball_result = detect_ball(ball_model, frame_copy)
        if ball_result and ball_result[0].boxes:
            valid_boxes = [box for box in ball_result[0].boxes if box.conf >= BALL_CONF_THRESHOLD]
            if valid_boxes:
                sorted_boxes = sorted(valid_boxes, key=lambda x: x.conf, reverse=True)
                best_box = sorted_boxes[0]
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_copy, "Ball", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        highlight_file = os.path.join(highlights_dir, f"highlight_frame_{idx}.jpg")
        cv2.imwrite(highlight_file, frame_copy)
    print(f"Highlights for ball {ball_number} generated and saved in {highlights_dir}")
    return highlights_dir

def create_Highlights_videos_from_frames(frames_folder, output_video_path, fps=30):

    image_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')],
                         key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    if not image_files:
        print("No frames found in", frames_folder)
        return
    first_frame = cv2.imread(os.path.join(frames_folder, image_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for img_file in image_files:
        frame_path = os.path.join(frames_folder, img_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video_writer.write(frame)
    video_writer.release()
    print("Video created at:", output_video_path)

def create_individual_ball_videos(video_output_directory,highlights_dir, ball_count, fps=30):
 
    ball_video_paths = []
    for b in range(1, ball_count + 1):
        ball_folder = os.path.join(video_output_directory, "highlights", f"ball_{b}")
        if os.path.exists(ball_folder):
            temp_video = os.path.join(highlights_dir, f"temp_ball_{b}_highlights.mp4")
            output_video = os.path.join(highlights_dir, f"ball_{b}_highlights.mp4")

            # Remove the output file if it already exists
            create_Highlights_videos_from_frames(ball_folder, temp_video, fps)
            # ball_video_paths.append(output_video)
            # Encode the video using ffmpeg
         # Verify that the temporary video file exists
            if not os.path.exists(temp_video):
                logging.error(f"Temporary video file not found: {temp_video}")
                continue

            # Encode the video using ffmpeg
            command = [
                'ffmpeg',
                '-y',
                '-i', temp_video, # Input file         
                '-c:v', 'libx264',         # Video codec
                '-profile:v', 'high',      # Video profile
                '-pix_fmt', 'yuv420p',     # Pixel format
                '-c:a', 'aac',             # Audio codec
                output_video,           # Output file
            ]

            try:
                subprocess.run(command, check=True)
                logging.debug(f"Video encoding completed successfully for ball {b}. Video saved as {output_video}")
                ball_video_paths.append(output_video)
            except subprocess.CalledProcessError as e:
                logging.error(f"An error occurred during video encoding for ball {b}: {e}")
            except FileNotFoundError as e:
                logging.error(f"FFmpeg not found: {e}")

            # Remove the temporary video file
            if os.path.exists(temp_video):
                os.remove(temp_video)

            

    return ball_video_paths

def new_func(output_video):
    return output_video


def merge_ball_videos(ball_video_paths, output_video_path, fps=30):

    temp_merged_video = os.path.join(os.path.dirname(output_video_path), "temp_merged_highlights.mp4")

    # Get frame size from first video
    cap = cv2.VideoCapture(ball_video_paths[0])
    ret, frame = cap.read()
    if not ret:
        print("Error reading the first video.")
        return
    height, width, _ = frame.shape
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_merged_video, fourcc, fps, (width, height))
    
    for video_path in ball_video_paths:
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    out.release()
     # Encode the merged video using ffmpeg
    command = [
        'ffmpeg',
        '-y',
        '-i', temp_merged_video,      # Input file
        '-c:v', 'libx264',            # Video codec
        '-profile:v', 'high',         # Video profile
        '-pix_fmt', 'yuv420p',        # Pixel format
        '-c:a', 'aac',                # Audio codec
        output_video_path             # Output file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Merged and encoded complete highlights video saved at: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during video encoding: {e}")

    # Remove the temporary merged video file
    if os.path.exists(temp_merged_video):
        os.remove(temp_merged_video)
    return output_video_path


def process_video(video_file, main_output_directory):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_file}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = fps if fps <= 30 else 30
    frame_count = 0

    # Dictionaries to store ball frames and pitch data
    ball_sequences = {}
    pitch_data = {}
    current_ball_frames = []
    new_ball_started = False
    missing_frames = 0
    MISSING_THRESHOLD = 10
    ball_number = 0

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_output_directory = os.path.join(main_output_directory, video_name)
    final_frames_directory = os.path.join(video_output_directory, "final_frames")
    os.makedirs(video_output_directory, exist_ok=True)
    os.makedirs(final_frames_directory, exist_ok=True)

    TARGET_CLASS_1 = "Cricket_pitch"
    TARGET_CLASS_2 = "score_bar"
 
    CONFIDENCE_THRESHOLD_Pitch = 0.75
    CONFIDENCE_THRESHOLD_ScoreBar = 0.75
    

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video: if a ball is in progress, store it.
            if new_ball_started and current_ball_frames:
                ball_number += 1
                ball_sequences[ball_number] = current_ball_frames.copy()
                detect_highlights(current_ball_frames, ball_model, video_output_directory, ball_number)
            break
        
        frame_count += 1
        if fps > 0 and frame_rate > 0:
            frame_interval = int(fps / frame_rate)
            if frame_interval > 0 and frame_count % frame_interval != 0:
                continue


        results = model(frame)
        class_1_present = any(model.names[int(pred.cls)] == TARGET_CLASS_1 and pred.conf > CONFIDENCE_THRESHOLD_Pitch 
                              for pred in results[0].boxes)
        class_2_present = any(model.names[int(pred.cls)] == TARGET_CLASS_2 and pred.conf > CONFIDENCE_THRESHOLD_ScoreBar 
                              for pred in results[0].boxes)

        if class_1_present and class_2_present:
            Highlights_start = True
            # Detection present; save frame and reset missing counter.
            desired_outputfile = os.path.join(final_frames_directory, f"desired_frame_{frame_count}.jpg")
            cv2.imwrite(desired_outputfile, frame)
            current_ball_frames.append(frame) 
            missing_frames = 0
            # _ = detect_highlights(current_ball_frames, ball_model, video_output_directory)
            if not new_ball_started:
                new_ball_started = True
                ball_number += 1
                # Extract pitch coordinates from this first detection frame.
                pitch_coords = pitch_coordinates(results)
                if pitch_coords and None not in pitch_coords:
                    pitch_data[ball_number] = pitch_coords
                    print(f"Ball {ball_number}: Pitch coordinates extracted: {pitch_coords}")
                    draw_pitch_length_annotations(frame, *pitch_coords)
                    cv2.imwrite(os.path.join(video_output_directory, f"ball_{ball_number}_pitch_frame.jpg"), frame)
                else:
                    print(f"Ball {ball_number}: Pitch coordinates not detected.")
                 # Immediately start highlights detection on this frame as soon as pitch and score_bar are detected.
                # _ = detect_highlights([frame], ball_model, video_output_directory)
        else:
            if new_ball_started:
                missing_frames += 1
                if missing_frames > MISSING_THRESHOLD:
                    # End current ball sequence.
                    ball_sequences[ball_number] = current_ball_frames.copy()
                    print(f"Ball {ball_number} sequence ended with {len(current_ball_frames)} frames.")
                    # write_ball_sequence_frames(current_ball_frames, ball_number, video_output_directory)
                    # Generate highlights for the full ball sequence (including frames after the initial detection).
                    detect_highlights(current_ball_frames, ball_model, video_output_directory, ball_number)
                    current_ball_frames = []
                    new_ball_started = False
                    missing_frames = 0



    cap.release()
    cv2.destroyAllWindows()

    total_frames_path = []
    bounce_results = {}
    for b_num, frames_seq in ball_sequences.items():
        if b_num in pitch_data:
            hit_class, frame_paths = process_ball_sequence_transition(frames_seq, b_num,video_name, video_output_directory, pitch_data[b_num])
            total_frames_path.extend(frame_paths) 
            bounce_results[b_num] = (hit_class)
            # bounce_results.append(hit_class)
        else:
            bounce_results[b_num] = None 


    #  Create individual ball highlight videos
    highlights_dir = rf"C:\Users\Administrator\Desktop\New folder\Implement_Shot_Classfication_model\App\dashboard\public\{video_name}"
    os.makedirs(highlights_dir, exist_ok=True)
    ball_video_paths = create_individual_ball_videos(video_output_directory, highlights_dir, ball_number, fps=30)
    print(ball_video_paths)
    
    # Merge individual ball videos into a complete highlights video

    Com_Highlighted_dir = rf"C:\Users\Administrator\Desktop\New folder\Implement_Shot_Classfication_model\App\dashboard\public\{video_name}"
    os.makedirs(Com_Highlighted_dir, exist_ok=True)
    output_video_path = os.path.join(f"{Com_Highlighted_dir}", f"{video_name}_highlights.mp4")
    output_video_path = merge_ball_videos(ball_video_paths, output_video_path, fps=30)
   
    # Convert the sequence of frames back into videos if the sequence is greater than 40 frames
    frame_ranges, video_count, video_path = convert_frames_to_videos(final_frames_directory, video_name, video_output_directory)
    return total_frames_path, bounce_results, ball_video_paths

# 2. Define a function to preprocess the video
def preprocess_video(video_path, num_frames=32, frame_size=224):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    sampling_interval = max(total_frames // num_frames, 1)  # Evenly sample frames

    while frame_count < num_frames:
        frame_id = frame_count * sampling_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # Jump to the frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size))
        frames.append(frame)
        frame_count += 1
    if frame_count < num_frames:
        padding = num_frames - total_frames
        last_frame = frames[-1]
        frames.extend([last_frame] * padding)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames were extracted from the video.")

    # Normalize frames using the processor's mean and std
    transform = Compose([
        ToTensor(),
        Normalize(mean=processor_load.image_mean, std=processor_load.image_std),
    ])
    frames = torch.stack([transform(frame) for frame in frames])  # Shape: (num_frames, C, H, W)
    return frames, total_frames




def classify_videos(video_name, main_output_directory):
    video_name = video_name + "_output"
    video_output_directory = os.path.join(main_output_directory, video_name.split('_output')[0])
    videos = [f for f in os.listdir(video_output_directory) if f.startswith(video_name) and f.endswith('.mp4')]
    videos.sort()
    
    results = []
    
    
    if not videos:
        logging.debug(f"No videos found for classification in {video_output_directory}")
        return results

    def process_video(video):
        video_path = os.path.join(video_output_directory, video)
        frames_VideoMAE, total_frame_count = preprocess_video(video_path, num_frames=32, frame_size=224)
        frames_VideoMAE = frames_VideoMAE.unsqueeze(0)
        frames_VideoMAE = frames_VideoMAE

        # logits_VideoMAE = run_inference(model_load_VideoMAE, frames_VideoMAE)
        logits_vivit = run_inference(model_load_vivit, frames_VideoMAE)
        # class_label_VideoMAE, predicted_class_VideoMAE, confidence_VideoMAE = get_predicted_class(model_load_VideoMAE, logits_VideoMAE)
        class_label_Vivit, predicted_class_Vivit, confidence_Vivit = get_predicted_class(model_load_vivit, logits_vivit)
        # logging.debug(f"Classification results for {video}: {class_label_VideoMAE} with confidence {confidence_VideoMAE}")
        logging.debug(f"Classification results for {video}: {class_label_Vivit} with confidence {confidence_Vivit}")

        # frame_ranges_file = os.path.join(video_output_directory, f"{video_name.split('_output')[0]}_frame_ranges.txt")
        # with open(frame_ranges_file, 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         if video in line:
        #             frame_range = line.split(":")[1].strip()
        #             print(frame_range)
        #             break

        results.append({
            "video": video,
            "predicted_class": predicted_class_Vivit,
            "class_label" : class_label_Vivit
            })
        # results.append({
        #     "video": video,
        #     "predicted_class": predicted_class_Vivit,
        #     "class_label" : class_label_Vivit
        #     })
   
        
    
    with ThreadPoolExecutor() as executor:
        executor.map(process_video, videos)
    return results
    


if __name__ == "__main__":
    video_file = r'2_ball_match.mp4'
    main_output_directory = "extracted_videos_output"
    os.makedirs(main_output_directory, exist_ok=True)

    try:
        frame_paths, bounce_results = process_video(video_file, main_output_directory)
        for ball, bounce in bounce_results.items():
            if bounce:
                classification = bounce
                print(f"Ball {ball}: Bounce Classification: {classification}")
            else:
                print(f"Ball {ball}: Bounce Classification could not be determined.")
        classify_videos(os.path.splitext(os.path.basename(video_file))[0], main_output_directory)
    except ValueError as e:
        print(f"Error: {e}")