import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from skimage.util import img_as_float, random_noise
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.feature import corner_peaks, corner_harris, canny, corner_shi_tomasi, peak_local_max


def img2gray(img):
    return img_as_float(rgb2gray(img))


def add_sp_noise(img, amount=0.05):
    img = img2gray(img)
    img = random_noise(img,
                       mode='s&p',
                       amount=amount)
    return img


def add_gaussian_noise(img, snr=20):
    img = img2gray(img)
    k = img.max() - img.min()
    deviation = k / 10 ** (snr / 20)
    img = random_noise(img,
                       mode='gaussian',
                       clip=True,
                       mean=0,
                       var=deviation ** 2)
    return img


def apply_gaussian_filter(img, sigma=3):
    img = gaussian(img,
                   sigma=sigma,
                   multichannel=True)
    return img


def corners_harris(img, method='k', k=0.05, eps=1e-06, sigma=1.,
                   min_distance=1, threshold_rel=0.1):
    img = img2gray(img)
    corners = corner_harris(img,
                            method=method,
                            k=k,
                            eps=eps,
                            sigma=sigma)
    corners = corner_peaks(corners,
                           min_distance=min_distance,
                           threshold_abs=None,
                           threshold_rel=threshold_rel,
                           exclude_border=True,
                           indices=True)
    return corners


def corners_shi_tomasi(img, sigma=1., min_distance=1, threshold_rel=0.1):
    img = img2gray(img)
    corners = corner_shi_tomasi(img, sigma)
    corners = corner_peaks(corners,
                           min_distance=min_distance,
                           threshold_abs=None,
                           threshold_rel=threshold_rel,
                           exclude_border=True,
                           indices=True)
    return corners


def edges_canny(img,  sigma=1., low_threshold=None, high_threshold=None,
                mask=None, use_quantiles=False):
    img = img2gray(img)
    edges = canny(img, sigma, low_threshold, high_threshold, mask, use_quantiles)
    edges_ind = np.array(np.nonzero(edges)).T
    return edges_ind


def show_pixel_coords(img, pixel_coords):
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title('Pixel corners')
    for k in range(len(pixel_coords)):
        [x, y] = pixel_coords[k]
        plt.plot(y,
                 x,
                 '+',
                 markersize=10,
                 color='white',
                 path_effects=[path_effects.withStroke(linewidth=3,
                                                       foreground='black')])
    plt.show(block=False)


def show_subpix_corners(detector):
    plt.figure()
    plt.imshow(detector.img, cmap='gray', vmin=0, vmax=1)
    plt.title('Subpixel corners coordinates')
    [k, n] = np.nonzero(detector.corners_map)
    for x, y in zip(k, n):
        if not np.isnan(detector.corners_map[x, y]):
            plt.plot(detector.y_p_img[x, y],
                     detector.x_p_img[x, y],
                     '+',
                     markersize=10,
                     color='white',
                     path_effects=[path_effects.withStroke(linewidth=3,
                                                           foreground='black')])
            plt.text(detector.y_p_img[x, y],
                     detector.x_p_img[x, y],
                     '({:.1f}, {:.1f})'.format(detector.y_p_img[x, y],
                                               detector.x_p_img[x, y]),
                     color='black',
                     path_effects=[path_effects.withStroke(linewidth=2,
                                                           foreground='white')])
    plt.show(block=False)


def show_subpix_edges(detector):
    plt.figure()
    plt.imshow(detector.img, cmap='gray', vmin=0, vmax=1)
    plt.title('Subpixel edges points')
    [k, n] = np.nonzero(detector.edges_map)
    for x, y in zip(k, n):
        if not np.isnan(detector.edges_map[x, y]):
            plt.plot(detector.y_p_img[x, y],
                     detector.x_p_img[x, y],
                     '.',
                     markersize=10,
                     color='white',
                     path_effects=[path_effects.withStroke(linewidth=3,
                                                           foreground='black')])
    plt.show(block=False)


def show_corners_angles(detector):
    plt.figure()
    plt.imshow(detector.img, cmap='gray', vmin=0, vmax=1)
    plt.title('Corners angles')
    [k, n] = np.nonzero(detector.corners_map)
    for x, y in zip(k, n):
        if not np.isnan(detector.corners_map[x, y]):
            plt.plot(detector.y_p_img[x, y],
                     detector.x_p_img[x, y],
                     '+',
                     markersize=10,
                     color='white',
                     path_effects=[path_effects.withStroke(linewidth=3,
                                                           foreground='black')])
            plt.text(detector.y_p_img[x, y],
                     detector.x_p_img[x, y],
                     '${:.1f}^\circ$'.format(np.rad2deg(detector.phi_img[x, y])*2),
                     color='black',
                     path_effects=[path_effects.withStroke(linewidth=2,
                                                           foreground='white')])
    plt.show(block=False)
