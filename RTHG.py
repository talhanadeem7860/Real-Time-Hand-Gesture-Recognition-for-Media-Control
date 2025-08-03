import cv2
import mediapipe as mp
import numpy as np
import math
import platform


# Import the correct library based on the operating system
OS_NAME = platform.system()

if OS_NAME == "Windows":
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    
    # Get default audio device
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    # Get volume range
    vol_range = volume.GetVolumeRange()
    MIN_VOL, MAX_VOL = vol_range[0], vol_range[1]
    print(f"OS Detected: Windows. Volume Range: {MIN_VOL} to {MAX_VOL}")

elif OS_NAME == "Darwin": 
    import subprocess
    
    def set_mac_volume(level):
        """Set volume for macOS using osascript."""
        subprocess.run(["osascript", "-e", f"set volume output volume {level}"])
        
    MIN_VOL, MAX_VOL = 0, 100
    print(f"OS Detected: macOS. Volume Range: {MIN_VOL} to {MAX_VOL}")
    
elif OS_NAME == "Linux":
    try:
        import pulsectl
        pulse = pulsectl.Pulse('volume-control')
        # Get the default sink (audio output)
        default_sink = pulse.sink_list()[0] 
        MIN_VOL, MAX_VOL = 0.0, 1.0 
        print(f"OS Detected: Linux (PulseAudio). Volume Range: {MIN_VOL} to {MAX_VOL}")
    except ImportError:
        print("OS Detected: Linux, but 'pulsectl' is not installed. Please run 'pip install pulsectl'.")
        exit()


# --- Hand Tracking Setup ---
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    # and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(image)

    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # List to store landmark coordinates
    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand annotations on the image.
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates of all landmarks
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])

    # If landmarks are detected, proceed with volume control logic
    if len(landmark_list) != 0:
        
        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]

        
        cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
       
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Calculate the distance between the two points
        length = math.hypot(x2 - x1, y2 - y1)
        
        # --- Map distance to volume ---
        # Hand distance range (can be adjusted based on your camera)
        min_dist = 20
        max_dist = 200
        
        # Map the length to the volume range for the current OS
        vol = np.interp(length, [min_dist, max_dist], [MIN_VOL, MAX_VOL])
        
        # For display purposes, calculate volume percentage
        vol_percent = np.interp(length, [min_dist, max_dist], [0, 100])

        # Set the system volume
        if OS_NAME == "Windows":
            volume.SetMasterVolumeLevel(vol, None)
        elif OS_NAME == "Darwin":
            set_mac_volume(vol)
        elif OS_NAME == "Linux":
            pulse.volume_set_all_chans(default_sink, vol)


        
        if length < min_dist:
            cv2.circle(image, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

       
        bar_x, bar_y, bar_w, bar_h = 50, 150, 85, 400
        vol_bar_height = int(np.interp(vol_percent, [0, 100], [bar_y + bar_h, bar_y]))
        
    
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 0, 0), 3)
       
        cv2.rectangle(image, (bar_x, vol_bar_height), (bar_x + bar_w, bar_y + bar_h), (255, 0, 0), cv2.FILLED)
       
        cv2.putText(image, f'{int(vol_percent)} %', (bar_x, bar_y - 20), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)


    # Display the resulting frame
    cv2.imshow('Hand Gesture Volume Control', image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()