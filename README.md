# Subpixel corner and edge detector

The proposed approach is based on the algebraic moments of the brightness function of halftone images. The ideal two-dimensional L-corner model is considered. The model has the following four parameters: the coordinates of the corner vertex, the orientation and the degree measure of the corner, and the brightness values from both sides of the corner. A particular case of the corner model is a linear model describing a linear edge. To obtain all subpixel parameters of the edge six algebraic moments are used [1]. To compute the moments rapidly masks are used. On the basis of the corner model an algorithm that makes it possible to quickly refine the coordinates of the corner and edge points on an image with subpixel accuracy is proposed. The use of integral characteristics increases the resistance to various kinds of noises.

# References
[1] Abramenko, A. A., and A. N. Karkishchenko. "Applications of algebraic moments for edge detection for locally linear model." Pattern Recognition and Image Analysis 27.3 (2017): 433-443.
