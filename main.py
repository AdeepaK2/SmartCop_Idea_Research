import cv2
import tkinter as tk
from tkinter import filedialog, Scale, IntVar, Label, Frame
import threading
import os
import numpy as np

# Function to detect vehicles in the video
def detect_vehicles(video_path, min_neighbors=2, scale_factor=1.1):
    # Use local cascade file
    cascade_path = os.path.join(os.path.dirname(__file__), 'cascades', 'haarcascade_car.xml')
    
    # Check if file exists
    if not os.path.isfile(cascade_path):
        print(f"Error: Cascade file not found at {cascade_path}")
        print("Please download the file first using the instructions in the comments.")
        return
        
    car_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Verify the classifier loaded correctly
    if car_cascade.empty():
        print("Error: Failed to load cascade classifier")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define counting lines - add two lines to make counting more reliable
    # One at 40% of height, one at 60%
    counting_line_y1 = int(height * 0.4)
    counting_line_y2 = int(height * 0.6)
    
    # Variables for vehicle counting
    total_vehicle_count = 0
    frame_count = 0
    
    # Dictionary to track vehicles
    tracked_vehicles = {}
    next_vehicle_id = 0
    
    # Improved parameters
    max_disappeared = 10  # Reduced from 15
    max_distance = 70     # Increased from 50

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Draw counting lines
        cv2.line(frame, (0, counting_line_y1), (width, counting_line_y1), (255, 0, 0), 2)
        cv2.line(frame, (0, counting_line_y2), (width, counting_line_y2), (0, 0, 255), 2)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track which vehicles we've seen in this frame
        current_detected_ids = set()
        
        # Detect cars every 3 frames (more frequent than before)
        if frame_count % 3 == 0:
            cars = car_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale_factor, 
                minNeighbors=min_neighbors,
                minSize=(50, 50)
            )
            
            # Debug - print number of cars detected
            if len(cars) > 0:
                print(f"Frame {frame_count}: Detected {len(cars)} cars")
            
            # Process detections
            for (x, y, w, h) in cars:
                if w > 50 and h > 50:  # Filter small detections
                    center_x = x + w//2
                    center_y = y + h//2
                    
                    matched = False
                    
                    # Try to match with existing vehicles
                    for vehicle_id, data in tracked_vehicles.items():
                        if data.get("disappeared", 0) > max_disappeared:
                            continue
                            
                        if data["center_positions"]:
                            last_cx, last_cy = data["center_positions"][-1]
                            
                            # Calculate distance between centers
                            distance = np.sqrt((center_x - last_cx)**2 + (center_y - last_cy)**2)
                            
                            # If close enough, it's the same vehicle
                            if distance < max_distance:
                                # Update tracking data
                                tracked_vehicles[vehicle_id]["box"] = (x, y, w, h)
                                tracked_vehicles[vehicle_id]["center_positions"].append((center_x, center_y))
                                tracked_vehicles[vehicle_id]["disappeared"] = 0
                                current_detected_ids.add(vehicle_id)
                                
                                # Check if vehicle crossed the counting line - count in BOTH directions
                                if len(data["center_positions"]) >= 2:
                                    prev_y = data["center_positions"][-2][1]
                                    curr_y = center_y
                                    
                                    # Count vehicles crossing either line in either direction
                                    if not data["counted"]:
                                        # Crossing first line (downward)
                                        if prev_y < counting_line_y1 and curr_y >= counting_line_y1:
                                            tracked_vehicles[vehicle_id]["counted"] = True
                                            total_vehicle_count += 1
                                            print(f"Vehicle {vehicle_id} counted! Total: {total_vehicle_count}")
                                        
                                        # Crossing second line (downward)
                                        elif prev_y < counting_line_y2 and curr_y >= counting_line_y2:
                                            tracked_vehicles[vehicle_id]["counted"] = True
                                            total_vehicle_count += 1
                                            print(f"Vehicle {vehicle_id} counted! Total: {total_vehicle_count}")
                                        
                                        # Crossing first line (upward)
                                        elif prev_y >= counting_line_y1 and curr_y < counting_line_y1:
                                            tracked_vehicles[vehicle_id]["counted"] = True
                                            total_vehicle_count += 1
                                            print(f"Vehicle {vehicle_id} counted! Total: {total_vehicle_count}")
                                        
                                        # Crossing second line (upward)
                                        elif prev_y >= counting_line_y2 and curr_y < counting_line_y2:
                                            tracked_vehicles[vehicle_id]["counted"] = True
                                            total_vehicle_count += 1
                                            print(f"Vehicle {vehicle_id} counted! Total: {total_vehicle_count}")
                                
                                matched = True
                                break
                    
                    # If no match found, create a new vehicle
                    if not matched:
                        tracked_vehicles[next_vehicle_id] = {
                            "box": (x, y, w, h),
                            "center_positions": [(center_x, center_y)],
                            "counted": False,
                            "disappeared": 0
                        }
                        current_detected_ids.add(next_vehicle_id)
                        next_vehicle_id += 1
        
        # Update disappeared counter for vehicles not found in this frame
        for vehicle_id in tracked_vehicles:
            if vehicle_id not in current_detected_ids:
                tracked_vehicles[vehicle_id]["disappeared"] = tracked_vehicles[vehicle_id].get("disappeared", 0) + 1
        
        # Remove vehicles that have been gone too long
        vehicles_to_remove = [vehicle_id for vehicle_id, data in tracked_vehicles.items() 
                             if data.get("disappeared", 0) > max_disappeared]
        for vehicle_id in vehicles_to_remove:
            tracked_vehicles.pop(vehicle_id, None)
        
        # Draw bounding boxes for active vehicles
        for vehicle_id, data in tracked_vehicles.items():
            if data.get("disappeared", 0) <= max_disappeared:
                x, y, w, h = data["box"]
                color = (0, 255, 0) if data["counted"] else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"ID: {vehicle_id}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1, cv2.LINE_AA)
        
        # Display counts on frame
        active_count = sum(1 for data in tracked_vehicles.values() if data.get("disappeared", 0) <= max_disappeared)
        text = f"Active vehicles: {active_count}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2, cv2.LINE_AA)
        
        total_text = f"Total vehicles counted: {total_vehicle_count}"
        cv2.putText(frame, total_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Vehicle Detection', frame)

        if cv2.waitKey(1) == 27:  # Press ESC to exit early
            break

    print(f"Total vehicles counted: {total_vehicle_count}")

    cap.release()
    cv2.destroyAllWindows()

# Function to select video and start detection
def select_video():
    video_path = filedialog.askopenfilename(title="Select Video",
                                            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
    if video_path:
        min_neighbors = min_neighbors_var.get()
        scale_factor = scale_factor_var.get() / 10.0
        threading.Thread(target=detect_vehicles, args=(video_path, min_neighbors, scale_factor)).start()

# GUI
root = tk.Tk()
root.title("Vehicle Detection App")
root.geometry("400x300")

# Create a frame for controls
control_frame = Frame(root, padx=10, pady=10)
control_frame.pack(fill="x")

# Detection parameters
min_neighbors_var = IntVar(value=2)
scale_factor_var = IntVar(value=11)  # We'll divide by 10 to get 1.1

# Parameter adjustment controls
Label(control_frame, text="Detection Sensitivity:").grid(row=0, column=0, sticky="w", pady=5)
Scale(control_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
      variable=min_neighbors_var, length=200).grid(row=0, column=1, pady=5)

Label(control_frame, text="Scale Factor:").grid(row=1, column=0, sticky="w", pady=5)
Scale(control_frame, from_=10, to=20, orient=tk.HORIZONTAL, 
      variable=scale_factor_var, length=200).grid(row=1, column=1, pady=5)

# Button to select video
btn = tk.Button(root, text="Select Video", command=select_video,
               bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5)
btn.pack(pady=20)

# Instructions
instructions = Label(root, text="Adjust sensitivity before selecting video.\nPress ESC to exit detection.", 
                    justify=tk.LEFT, padx=10, pady=5)
instructions.pack(pady=10)

root.mainloop()
