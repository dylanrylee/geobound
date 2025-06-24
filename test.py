import cv2
import numpy as np

# Regenerate the outlined image
img_path = '/mnt/data/09ab62f0-0314-4566-90de-71d442e111e4.png'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)
blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outlined = img.copy()
cv2.drawContours(outlined, contours, -1, (0, 255, 0), 2)

# Save the result
output_path = '/mnt/data/outlined_parcels.png'
cv2.imwrite(output_path, outlined)
print(f"Outlined image saved to {output_path}")
