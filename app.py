import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

model = YOLO("runs_v8s/train/train/weights/best.onnx", task="detect")
print(f'MOdel.names: {model.names}')

# The list of the labels that we filtered/limited the data to
filter_list = [
    'Bottle cap',
    'Bottle',
    'Can',
    'Carton',
    'Cup',
    'Paper',
    'Straw',
    'Unlabeled litter',
]

# Initialize session state for All_Categories_that_appeared
@st.cache_resource
def get_empty_list():
    return []
#create a web-based interface for interacting with the mode
@st.cache_resource
#function processes an image using the YOLO model to detect objects. It updates the list of detected labels,

def load_model(model_path: str = None):
    '''
    Function to load the model (the fine-tuned model)
    '''
    if model_path is None:
        raise ValueError("No model path provided")
    else:
        model = YOLO(model_path, task='detect')
        print(model)
        global class_counts
        return model
#function counts how many instances of each label (object type) are present in the image.
def infer(image: np.ndarray, model, All_Cats):
    '''
    Function to infer the model i.e. get the results on a frame
    '''
    results = model(image)

    Labels_in_frame = []
    for result in results:
        try:
            Labels_in_frame.append(int(result.boxes.cls.cpu()[0].item()))
        except:
            pass
        
    class_counts = {}
    class_counts.update({model.names[i]: 0 for i in range(18)})

    # Remove the labels that are not in the filter list
    copy_class_counts = class_counts.copy()
    for key in class_counts.keys():
        if key not in filter_list:
            copy_class_counts.pop(key)
    
    # Update the class counts
    for label in Labels_in_frame:
        copy_class_counts[model.names[label]] += 1
        if model.names[label] not in All_Cats:
            All_Cats.append(model.names[label])
            print(f'\n\nnew session_state = {All_Cats}\n\n')

    # Create a blank image to display the counts
    with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
        results[0].save(f.name)
        results_frame = cv2.imread(f.name)
        
    count_img = results_frame

    # Display the counts on the top right of the frame
    for i, (label, count) in enumerate(copy_class_counts.items()):
        text = f'{label}: {count}'
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Define position and size for the black box
        box_coords = (
            (image.shape[1] - 210, 20 + i * 20),
            (image.shape[1] - 200 + text_width + 10, 30 + i * 20 + text_height)
        )
        
        # Draw the black box
        cv2.rectangle(count_img, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
        
        # Draw the text with a black outline
        cv2.putText(count_img, text, (image.shape[1] - 200, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        
        # Draw the text in white on top of the black outline
        cv2.putText(count_img, text, (image.shape[1] - 200, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Combine the count image with the original frame
    output_frame = cv2.addWeighted(image, 1, count_img, 1, 0)

    return output_frame, All_Cats

def gmail_create_draft(body):
    msg = MIMEMultipart()
    msg['From'] = "streetwastemanage@gmail.com"
    msg['To'] = "streetwastemanage@gmail.com"
    msg['Subject'] = 'Trash Detection Results'
    
    # Use All_Categories_that_appeared from session state
    print(f'body: {body}')
    
    if not body:
        print("Warning: No categories detected!")
        return False
    
    body_text = f'''The trash categories that have been detected are: {','.join(body)}'''

    msg.attach(MIMEText(body_text, 'plain'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(msg['From'], 'ousl yntp hezy iiah')
            text = msg.as_string()
            
            server.sendmail(msg['From'], msg['To'], text)
            print("Email sent successfully")
            return True
    except Exception as e:
        print(f"Error: unable to send email. Details: {e}")
        return False

st.title("Webcam Live Feed")
All_Cats = get_empty_list()

# Button to start or stop video feed
if 'video_feed_active' not in st.session_state:
    st.session_state['video_feed_active'] = True

# Button to send email
if 'email_sent' not in st.session_state:
    st.session_state['email_sent'] = False

# Display the video feed
if st.button("Start Video Feed"):
    st.session_state['video_feed_active'] = True

if st.button("Stop Video Feed"):
    st.session_state['video_feed_active'] = False

if st.session_state['video_feed_active']:
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    frame_idx = 0
    while st.session_state['video_feed_active']:
        _, frame = camera.read()
        frame_idx += 1

        if frame_idx % 1 == 0:  # Skip frames if needed, currently skipping none
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame)
            res, All_Cats = infer(frame, model,All_Cats)
            FRAME_WINDOW.image(res)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame)
            FRAME_WINDOW.image(frame)

    camera.release()
    cv2.destroyAllWindows()

# Button to send email
if st.button("Send Email") and not st.session_state['email_sent']:
    if gmail_create_draft(body=All_Cats):
        st.success("Email draft created successfully!")
    else:
        st.error("Failed to create email draft.")
    st.session_state['email_sent'] = True
