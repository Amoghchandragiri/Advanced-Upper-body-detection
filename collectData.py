import cv2
import mediapipe as mp
import math
import os


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


data_folder = "data"
os.makedirs(data_folder, exist_ok=True)


box_width = 170
box_height = 400

# Define the body parts
body_parts = ["hip", "shoulder", "upper_body", "leg"]

# Prompt the user to enter the number of persons
num_persons = int(input("Enter the number of persons: "))


measurements_file = os.path.join(data_folder, 'measurements.txt')
mediapipe_data_file = os.path.join(data_folder, 'mediapipe_data.txt')

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark2.x - landmark1.x)**2 + (landmark2.y - landmark1.y)**2)

# Function to check visibility of landmarks
def is_visible(landmark, threshold=0.5):
    return landmark.visibility >= threshold

# Open the measurements file in append mode
with open(measurements_file, 'a') as file, open(mediapipe_data_file, 'a') as mediapipe_file:
    # Loop through each person
    for person in range(1, num_persons + 1):
        # Prompt the user to enter real measurements for all body parts
        print(f"Enter real measurements for Person {person}:")
        measurements = {}
        for body_part in body_parts:
            measurement = input(f"{body_part}: ")
            measurements[body_part] = measurement

        # Write the measurements to the measurements file
        for body_part, measurement in measurements.items():
            file.write(f"{measurement}\n")

        # Open the video capture
        cap = cv2.VideoCapture(0)

        # Check if the video capture is successfully opened
        if not cap.isOpened():
            raise IOError("Failed to open video capture.")

        # Start the upper body estimation process
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            measurement_data = []
            while cap.isOpened():
                # Read a frame from the video capture
                ret, frame = cap.read()
                if not ret:
                    continue

                # Flip the frame horizontally for a mirror effect
                frame = cv2.flip(frame, 1)

                # Convert the frame from BGR to RGB for processing with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run pose estimation on the frame
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Extract specific landmarks for measurements
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

                    # Debugging: Check if landmarks are detected
                    print(f"Left Shoulder Visibility: {left_shoulder.visibility}, Left Hip Visibility: {left_hip.visibility}")

                    # Calculate upper body distance only if both landmarks are visible
                    if is_visible(left_shoulder) and is_visible(left_hip):
                        upper_body_distance = calculate_distance(left_shoulder, left_hip)
                        print(f"Upper Body Distance: {upper_body_distance}")  # Debugging line
                    else:
                        upper_body_distance = None

                    # Calculate shoulder distance only if both shoulders are visible
                    if is_visible(left_shoulder) and is_visible(right_shoulder):
                        shoulder_distance = calculate_distance(left_shoulder, right_shoulder)
                    else:
                        shoulder_distance = None

                    # Draw a bounding box on the frame
                    frame_height, frame_width, _ = frame.shape
                    box_left = int((frame_width - box_width) / 2)
                    box_top = int((frame_height - box_height) / 2)
                    box_right = box_left + box_width
                    box_bottom = box_top + box_height
                    cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)

                    # Display only the upper body and shoulder measurements
                    if upper_body_distance is not None:
                        cv2.putText(frame, f"upper_body: {upper_body_distance:.2f} pixels", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "upper_body: Not Detected", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if shoulder_distance is not None:
                        cv2.putText(frame, f"shoulder: {shoulder_distance:.2f} pixels", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "shoulder: Not Detected", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Add measurements to the list if they exist
                    if upper_body_distance is not None and shoulder_distance is not None:
                        measurement_data.append((shoulder_distance, upper_body_distance))

                # Draw the upper body landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display the frame
                cv2.imshow('MediaPipe Pose Estimation', frame)

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save the best accuracy measurement
            if measurement_data:
                best_accuracy_measurement = max(measurement_data, key=lambda x: x[0])
                mediapipe_file.write(f"{best_accuracy_measurement[0]}\n")
                mediapipe_file.write(f"{best_accuracy_measurement[1]}\n")

        # Release the video capture
        cap.release()
        cv2.destroyAllWindows()
