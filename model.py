from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import PIL

class Model:

    def __init__(self):
        self.model = LinearSVC(max_iter=10000)

    def train_model(self, counters):
        img_list = []
        class_list = []

        for i in range(1, counters[0]):
            img_path = f'1/frame{i}.jpg'
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Skipping missing file {img_path}")
                continue

            print(f"Image {img_path} shape: {img.shape}")  # Debugging
            img = img.reshape(-1)  # Flatten dynamically
            img_list.append(img)
            class_list.append(1)

        for i in range(1, counters[1]):
            img_path = f'2/frame{i}.jpg'
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Skipping missing file {img_path}")
                continue

            print(f"Image {img_path} shape: {img.shape}")  # Debugging
            img = img.reshape(-1)
            img_list.append(img)
            class_list.append(2)

        if len(img_list) == 0:
            print("Error: No valid images found for training!")
            return

        img_list = np.array(img_list)
        class_list = np.array(class_list)
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")
    
    def predict(self, frame):
        if not hasattr(self.model, "coef_"):  # Check if model is trained
            print("Error: Model is not trained yet. Train the model first!")
            return None

        frame = frame[1]
        if frame is None:
            print("Error: No frame captured")
            return None

        img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        img = cv.resize(img, (150, 113))  # Resize to ensure consistent shape
        img = img.reshape(-1)

        try:
            prediction = self.model.predict([img])
            return prediction[0]
        except:
            print("Error: Model is not trained yet. Train the model first!")
            return None

    