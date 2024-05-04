# import cv2
# import mediapipe as mp
# import csv
# import os
# import uuid

# # Initialize VideoCapture object to capture video from the webcam
# cap = cv2.VideoCapture(0)

# # Initialize mediapipe hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Create a directory to store captured images
# image_dir = 'captured_images'
# os.makedirs(image_dir, exist_ok=True)

# # Open a CSV file to write finger count, ID number, and unique name
# csv_file = 'finger_data.csv'
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['ID', 'Name', 'Finger Count', 'Image Path'])

#     while True:
#         # Read a frame from the webcam
#         success, img = cap.read()
#         if not success:
#             print("Failed to read from webcam")
#             break

#         # Flip the image horizontally
#         img = cv2.flip(img, 1)

#         # Convert the image to RGB format
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Process the image to detect hands
#         results = hands.process(img_rgb)

#         finger_count = 0

#         if results.multi_hand_landmarks:
#             # For each detected hand
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Count the number of fingers based on the positions of landmarks
#                 # You can implement your finger counting algorithm here
#                 # This is just a placeholder implementation
#                 finger_count += 1  # Placeholder for counting fingers

#                 # Capture image and save
#                 unique_name = str(uuid.uuid4())[:8]  # Generate a unique name
#                 image_path = os.path.join(image_dir, f'{unique_name}.jpg')
#                 cv2.imwrite(image_path, img)

#                 # Write the ID, unique name, finger count, and image path to the CSV file
#                 writer.writerow([unique_name, unique_name, finger_count, image_path])

#         # Display the annotated image
#         cv2.imshow("Hand Landmarks", img)

#         # Break the loop when 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
