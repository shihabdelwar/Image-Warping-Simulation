import cv2
import numpy as np

def swirl_effect(image, strength=2, radius=200):
    """
    Applies a swirl distortion to an image, mimicking AIWS spatial warping.
    :param image: Input image.
    :param strength: Degree of swirl effect (higher = more distortion).
    :param radius: The area of the swirl effect.
    :return: Distorted image with swirl effect.
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Create meshgrid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x - center_x, y - center_y  # Centering
    r = np.sqrt(x**2 + y**2)

    # Swirl transformation
    theta = np.arctan2(y, x) + strength * np.exp(-r / radius)
    x_new = center_x + r * np.cos(theta)
    y_new = center_y + r * np.sin(theta)

    # Map coordinates
    map_x = np.clip(x_new, 0, w - 1).astype(np.float32)
    map_y = np.clip(y_new, 0, h - 1).astype(np.float32)

    # Apply transformation
    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return distorted_image

def fisheye_effect(image, strength=2):
    """
    Applies a fisheye distortion effect, making objects appear bulging or stretched.
    :param image: Input image.
    :param strength: Strength of the fisheye effect (0.1 - 1.0 recommended).
    :return: Distorted image with fisheye effect.
    """
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

    r = np.sqrt(map_x**2 + map_y**2)
    theta = np.arctan2(map_y, map_x)
    r = r**(1 + strength)  # Exaggerate distances

    map_x_new = w // 2 + w // 2 * r * np.cos(theta)
    map_y_new = h // 2 + h // 2 * r * np.sin(theta)

    map_x_new = np.clip(map_x_new, 0, w - 1).astype(np.float32)
    map_y_new = np.clip(map_y_new, 0, h - 1).astype(np.float32)

    distorted_image = cv2.remap(image, map_x_new, map_y_new, interpolation=cv2.INTER_LINEAR)
    return distorted_image

# Load an image (replace 'your_image.jpg' with your file path)
image = cv2.imread("Downloads/IMG_0765.JPG")
if image is None:
    print("Error: Image not found.")
    exit()

# Apply distortions
swirled = swirl_effect(image)
fisheye = fisheye_effect(image)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Swirl Effect", swirled)
cv2.imshow("Fisheye Effect", fisheye)

# Clean up and close the window
cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()

