import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os   
import time

# Function to extract HOG features from an image
def extract_hog_features(image):
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(image)
    features = features.flatten()
    return features


# Record the start time
start_time = time.time()

# Load and preprocess images
# Define paths to the folders containing manure and no manure images
# manure_folder = "D:\manure-detection\manure_folder"
# no_manure_folder = "D:/manure-detection/no_manure_folder"
manure_folder = "C:\\Users\\roger\\OneDrive\\Desktop\\manure-detection\\manure_folder"
no_manure_folder = "C:\\Users\\roger\\OneDrive\\Desktop\\manure-detection\\no_manure_folder"
# Initialize lists to store image paths and labels
image_paths_manure = []
image_paths_no_manure = []
X = []
y = []

# Iterate over the images in the manure folder
for filename in os.listdir(manure_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter only image files
        image_path = os.path.join(manure_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features = extract_hog_features(gray_image)
            X.append(features)
        y.append(1)  # Label 1 for manure

# Iterate over the images in the no manure folder
for filename in os.listdir(no_manure_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter only image files
        image_path = os.path.join(no_manure_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features = extract_hog_features(gray_image)
            X.append(features)
        y.append(0)  # Label 0 for no manure

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X)
# print('with manure: ',image_paths_manure)
# print('with no manure: ',image_paths_no_manure)

# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predict on test set
y_pred = svm_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Your existing code block here

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# # Test on a sample image
# test_img_path = "test_image.jpg"
# test_img = cv2.imread(test_img_path)
# test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# test_features = extract_hog_features(test_img_gray)
# test_features = np.array(test_features).reshape(1, -1)
# prediction = svm_classifier.predict(test_features)

# if prediction == 1:
#     print("Manure detected in the test image!")
# else:
#     print("No manure detected in the test image.")
