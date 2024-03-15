import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Load and preprocess images
# In this example, let's assume you have a list of image paths
image_paths_manure = ["manure1.jpg", "manure2.jpg", ...]
image_paths_no_manure = ["no_manure1.jpg", "no_manure2.jpg", ...]
X = []
y = []

# Load manure images
for path in image_paths_manure:
    image = cv2.imread(path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = extract_hog_features(image_gray)
    X.append(features)
    y.append(1)  # Label 1 for manure

# Load background images without manure
for path in image_paths_no_manure:
    image = cv2.imread(path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = extract_hog_features(image_gray)
    X.append(features)
    y.append(0)  # Label 0 for no manure

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predict on test set
y_pred = svm_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test on a sample image
test_img_path = "test_image.jpg"
test_img = cv2.imread(test_img_path)
test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
test_features = extract_hog_features(test_img_gray)
test_features = np.array(test_features).reshape(1, -1)
prediction = svm_classifier.predict(test_features)

if prediction == 1:
    print("Manure detected in the test image!")
else:
    print("No manure detected in the test image.")
