# main.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Load Images ---
# Get the list of image files, sorted to ensure order
image_folder = 'images'
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])

# Load the first two images
img1 = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image_files[1], cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: Could not load one or more images.")
else:
    print("Images loaded successfully.")

    # --- 2. Initialize SIFT Detector ---
    sift = cv2.SIFT_create()

    # --- 3. Find Keypoints and Descriptors ---
    # kp will be the list of keypoints, des will be the numpy array of descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print(f"Found {len(kp1)} keypoints in Image 1 and {len(kp2)} keypoints in Image 2.")

    # --- 4. Match Keypoints ---
    # Use BFMatcher (Brute-Force Matcher) with default params
    bf = cv2.BFMatcher()
    # Find the 2 best matches for each descriptor (k=2)
    matches = bf.knnMatch(des1, des2, k=2)
    print(f"Found {len(matches)} initial matches.")

    # --- 5. Apply Lowe's Ratio Test to find good matches ---
    # This is a standard method to filter out ambiguous matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    print(f"Found {len(good_matches)} good matches after ratio test.")

    # --- 6. Visualize the Matches ---
    # cv2.drawMatches needs a list of lists for single matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Use Matplotlib to display the image
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title('Good Matches Between Image 1 and Image 2')
    plt.show()



if len(good_matches) > 10: # We need at least a few matches to proceed
    # --- 7. Find the Essential Matrix ---
    
    # Get the coordinates of the good matches
    pts1 = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    pts2 = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    # For the Essential Matrix, we need the camera's intrinsic parameters.
    # We will assume a simple, idealized camera for now.
    # The focal length is typically guessed as the image width.
    focal_length = img1.shape[1]
    center_point = (img1.shape[1]/2, img1.shape[0]/2)
    camera_matrix = np.array([[focal_length, 0, center_point[0]],
                              [0, focal_length, center_point[1]],
                              [0, 0, 1]])

    # Calculate the Essential Matrix using RANSAC
    E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # The 'mask' tells us which of our 'good_matches' were kept by RANSAC (the inliers)
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
    print(f"Found {len(inlier_matches)} inlier matches after RANSAC.")

    # --- 8. Visualize the Inlier Matches ---
    img_inlier_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(20, 10))
    plt.imshow(img_inlier_matches)
    plt.title('Inlier Matches (after RANSAC)')
    plt.show()

else:
    print("Not enough good matches were found to proceed.")

# --- 9. Recover the camera's pose (Rotation and Translation) ---
# We use the inlier points from the RANSAC step for this
inlier_pts1 = np.float32([ kp1[m.queryIdx].pt for m in inlier_matches ]).reshape(-1,1,2)
inlier_pts2 = np.float32([ kp2[m.trainIdx].pt for m in inlier_matches ]).reshape(-1,1,2)

# The recoverPose function returns the number of inliers which passed the check,
# the rotation matrix (R), the translation vector (t), and a new mask for points
points, R, t, mask = cv2.recoverPose(E, inlier_pts1, inlier_pts2, camera_matrix)

print("\n--- Camera Pose Recovered ---")
print(f"Rotation Matrix (R):\n{R}")
print(f"\nTranslation Vector (t):\n{t}")

# --- 10. Triangulate 3D points ---
# We need to create the projection matrices for each camera
# The first camera is at the origin, so its projection matrix is simple
P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
# The second camera's projection matrix is derived from R and t
P2 = np.hstack((R, t))

# We need to use the camera matrix for triangulation
P1 = camera_matrix @ P1
P2 = camera_matrix @ P2

# Triangulate the points
# The output is in 4D homogeneous coordinates
points_4d = cv2.triangulatePoints(P1, P2, inlier_pts1, inlier_pts2)

# Convert from homogeneous to 3D Cartesian coordinates
points_3d = points_4d / points_4d[3]
# Transpose to get a (N, 3) shape
points_3d = points_3d[:3, :].T

print(f"\nSuccessfully triangulated {len(points_3d)} points.")


# --- 11. Visualize the 3D Point Cloud ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the 3D points
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=points_3d[:, 2], cmap='viridis', marker='o')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Reconstruction Point Cloud')
# Set a good viewing angle
ax.view_init(elev=-75, azim=-90)

plt.show()