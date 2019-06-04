# Subpixel corner and edge detector

The proposed approach is based on the algebraic moments of the brightness function of halftone images. The ideal two-dimensional L-corner model is considered. The model has the following four parameters: the coordinates of the corner vertex, the orientation and the degree measure of the corner, and the brightness values from both sides of the corner. A particular case of the corner model is a linear model describing a linear edge. To obtain all subpixel parameters of the edge six algebraic moments are used. To compute the moments rapidly masks are used. On the basis of the corner model an algorithm that makes it possible to quickly refine the coordinates of the corner and edge points on an image with subpixel accuracy is proposed. The use of integral characteristics increases the resistance to various kinds of noises.

[Examples](https://github.com/Abramenko/subpixel_corner_edge_detector/blob/master/img/results/README.md)

# References
[1] Abramenko A.A., Karkishchenko A.N. "Applications of algebraic moments for edge detection for locally linear model." Pattern Recognition and Image Analysis 27.3 (2017): 433-443.

[2] Abramenko A.A., Karkishchenko A.N. "Applications of algebraic moments for corner and edge detection for locally angular model." Pattern Recognition and Image Analysis 29.1 (2019): 58-71.

# Citation
Use this bibtex to cite:

```
@Article{abramenko2017applications,
  author    = {Abramenko, A. A. and Karkishchenko, A. N.},
  title     = {Applications of algebraic moments for edge detection for locally linear model},
  journal   = {Pattern Recognit. Image Anal.},
  year      = {2017},
  volume    = {27},
  number    = {3},
  pages     = {433-443},
  issn      = {1054-6618},
  doi       = {10.1134/S1054661817030026},
  url       = {http://link.springer.com/10.1134/S1054661817030026},
  publisher = {Pleiades Publishing},
}
```
```
@Article{abramenko2019applications,
  author    = {Abramenko, A. A. and Karkishchenko, A. N.},
  title     = {Applications of algebraic moments for corner and edge detection for locally angular model},
  journal   = {Pattern Recognit. Image Anal.},
  year      = {2019},
  volume    = {29},
  number    = {1},
  pages     = {58-71},
  issn      = {1555-6212},
  doi       = {10.1134/S1054661819010024},
  url       = {https://doi.org/10.1134/S1054661819010024},
  publisher = {Pleiades Publishing},
}
```