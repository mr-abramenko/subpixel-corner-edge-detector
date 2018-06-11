import numpy as np
import matplotlib.pyplot as plt
from utils import img2gray, add_sp_noise, add_gaussian_noise, apply_gaussian_filter, corners_harris, corners_shi_tomasi, edges_canny, show_subpix_corners, show_subpix_edges, show_corners_angles, show_pixel_coords
from subpixel_corner_edge_detector import *

# ----------
# Parameters
# ----------
img_path = 'img/test.png'
# img_path = 'img/test.jpg'

d = 25          # diameter (d=2*R) of neighborhood (domain: > 0, restriction: odd integer)
h_c = 0.05      # threshold for measure of heterogeneity (domain: [0, R])
h_s = 0.75      # threshold for measure of symmetry (domain: [0, 1])
h_ab = 0.05     # threshold for contrast (domain: [0, 1])
h_c_hat = 0.1   # threshold for the measure of the corner (domain: [0, 1])
h_e_hat = 0.95  # threshold for the measure of the edge (domain: [0, 1])

sp_noise_amount = 0     # amount of salt-paper noise (domain: [0, 1])
gaus_noise_snr = 15      # signal-to-noise ratio for gaussian noise (domain: >= 0)
sigma = 3               # sigma for gaussian filter (domain: >= 1)
# ----------

# Get image
img_orig = img2gray(plt.imread(img_path))
img = img_orig.copy()

# Add noise
if sp_noise_amount:
    img = add_sp_noise(img, amount=sp_noise_amount)
if gaus_noise_snr:
    img = add_gaussian_noise(img, snr=gaus_noise_snr)
# if sigma:
#     img = apply_gaussian_filter(img, sigma=sigma)

# Show image
plt.figure()
plt.subplot(1, 2, 1)
plt.title('Original image')
plt.imshow(img_orig, cmap='gray', vmin=0, vmax=1)
plt.subplot(1, 2, 2)
plt.title('Noisy image')
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.show(block=False)

# Corners detection
pixel_corners_coords = corners_harris(img, sigma=sigma, min_distance=1, threshold_rel=0.05)
# pixel_corners_coords = corners_shi_tomasi(img, sigma=sigma, min_distance=1, threshold_rel=0.05)
detector = SubpixelCornerEdgeDetector(img, pixel_corners_coords, d)
subpixel_corners_coords = detector.detect_corners(h_ab, h_c, h_s, h_c_hat, min_dist=d)

show_pixel_coords(img, pixel_corners_coords)
show_subpix_corners(detector)
show_corners_angles(detector)

# Edges detection
pixel_coords = edges_canny(img, sigma=sigma)
detector = SubpixelCornerEdgeDetector(img, pixel_coords, d)
subpixel_coords = detector.detect_edges(h_ab, h_c, h_s, h_e_hat, min_dist=0)

show_subpix_edges(detector)
