import numpy as np
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.feature import corner_peaks


class CircImgMoments:
    """
    Class that implements fast computations of moments
    of various orders inside a circular neighborhood (circular window) [1]

    [1] Abramenko, A. A., and A. N. Karkishchenko. "Applications of algebraic
     moments for edge detection for locally linear model." Pattern Recognition
     and Image Analysis 27.3 (2017): 433-443.
    """

    def __init__(self, d=15):
        assert (d % 2 != 0) and isinstance(d, int), "'d' must be odd integer"

        self.d = d

        self.mask00 = self.get_mask(0, 0, self.d)
        self.mask10 = self.get_mask(1, 0, self.d)
        self.mask01 = self.get_mask(0, 1, self.d)
        self.mask11 = self.get_mask(1, 1, self.d)
        self.mask20 = self.get_mask(2, 0, self.d)
        self.mask02 = self.get_mask(0, 2, self.d)
        self.mask21 = self.get_mask(2, 1, self.d)
        self.mask12 = self.get_mask(1, 2, self.d)
        self.mask30 = self.get_mask(3, 0, self.d)
        self.mask03 = self.get_mask(0, 3, self.d)

    @classmethod
    def get_mask(cls, p, q, d):
        def integral_I(a, b):
            if 0 <= p+q <= 3:  # if necessary, it can be extended for higher order moments
                if p == 0 and q == 0:
                    I_00 = lambda x: (x/2 * (r**2 - x**2)**(1/2) +
                                      r**2 / 2 * np.arcsin(x/r))
                    I = I_00(b) - I_00(a)
                elif p == 1 and q == 0:
                    I_10 = lambda x: -1/3 * (r**2 - x**2)**(3/2)
                    I = I_10(b) - I_10(a)
                elif p == 0 and q == 1:
                    I_01 = lambda x: r**2 * x - x**3 / 3
                    I = I_01(b) - I_01(a)
                elif p == 1 and q == 1:
                    I_11 = lambda x: r**2 * x**2 / 2 - x**4 / 4
                    I = I_11(b) - I_11(a)
                elif p == 2 and q == 0:
                    I_20 = lambda x: (x/8 * (2*x**2 - r**2) *
                                      (r**2 - x**2)**(1/2) +
                                      r**4 / 8 * np.arcsin(x/r))
                    I = I_20(b) - I_20(a)
                elif p == 0 and q == 2:
                    I_02 = lambda x: (
                            x/4 * (r**2 - x**2)**(3/2) +
                            3 * r**2 * x/8 * (r**2 - x**2)**(1/2) +
                            3 * r**4 / 8 * np.arcsin(x/r))
                    I = I_02(b) - I_02(a)
                elif p == 1 and q == 2:
                    I_12 = lambda x: -1/5 * (r**2 - x**2)**(5/2)
                    I = I_12(b) - I_12(a)
                elif p == 2 and q == 1:
                    I_21 = lambda x: r**2 * x**3 / 3 - x**5 / 5
                    I = I_21(b) - I_21(a)
                elif p == 3 and q == 0:
                    I_30 = lambda x: (1/5 * (r**2 - x**2)**(5/2) -
                                      r**2 / 3 * (r**2 - x**2)**(3/2))
                    I = I_30(b) - I_30(a)
                elif p == 0 and q == 3:
                    I_03 = lambda x: (r**4 * x -
                                      2*r**2 * x**3 / 3 + x**5 / 5)
                    I = I_03(b) - I_03(a)
                return I
            else:
                raise TypeError("integral_I defined for 0 <= p+q <= 3")

        if not (isinstance(p, int) and isinstance(q, int)):
            raise TypeError("'p' and 'q' must be integers")
        if (d%2 != 0) and isinstance(d, int):
            r = d/2
        else:
            raise TypeError("'d' must be odd integer")

        m_pq = np.zeros((d, d))
        r_ind = int(np.fix(r))
        for i1 in range(r_ind + 1):
            for j1 in range(r_ind + 1):
                if (i1 - 1/2)**2 + (j1 - 1/2)**2 > r**2:
                    m_pq[r_ind - j1, r_ind + i1] = 0
                elif (i1 + 1/2)**2 + (j1 + 1/2)**2 < r**2:
                    m_pq[r_ind - j1, r_ind + i1] = (
                        ((i1 + 1/2)**(p+1) - (i1 - 1/2)**(p+1)) / (p+1)
                        * ((j1 + 1/2)**(q+1) - (j1 - 1/2)**(q+1)) / (q+1))
                else:
                    if (i1 == 0) and (j1 == r - 1/2):
                        m_pq[r_ind - j1, r_ind + i1] = (
                            (1 / (q+1)) *
                            integral_I(-1/2, 1/2) -
                            ((r - 1)**(q+1) / ((p+1)*(q+1)*2**(p+1))) *
                            (1 - (-1)**(p+1))
                            )
                    elif (j1 == 0) and (i1 == r - 1/2):
                        m_pq[r_ind - j1, r_ind + i1] = (
                            (1 - (-1)**(q+1)) / (q+1) *
                            (integral_I(np.sqrt(r**2 - 1/4), r) +
                             (((r**2 - 1/4)**((p+1)/2) -
                              (r - 1)**(p+1)) /
                             ((p+1) * 2**(q+1))))
                            )
                    else:
                        if ((j1 - 1/2) < np.sqrt(r**2 - (i1 - 1/2)**2)
                                < (j1 + 1/2) and (i1 - 1/2) <
                                np.sqrt(r**2 - (j1 - 1/2)**2)
                                < (i1 + 1/2)):
                            m_pq[r_ind - j1, r_ind + i1] = (
                                (1 / (q+1)) *
                                integral_I((i1 - 1/2),
                                           np.sqrt(r**2 - (j1 - 1/2)**2)) -
                                ((j1 - 1/2)**(q+1) / ((p+1)*(q+1))) *
                                ((r**2 - (j1 - 1/2)**2)**((p+1)/2) -
                                 (i1 - 1/2)**(p+1))
                                )
                        elif ((j1 - 1/2) < np.sqrt(r**2 - (i1 - 1/2)**2)
                                < (j1 + 1/2) and (j1 - 1/2) <
                                np.sqrt(r**2 - (i1 + 1/2)**2)
                                < (j1 + 1/2)):
                            m_pq[r_ind - j1, r_ind + i1] = (
                                (1 / (q+1)) *
                                integral_I((i1 - 1/2), (i1 + 1/2)) -
                                ((j1 - 1/2)**(q+1) / ((p+1)*(q+1))) *
                                ((i1 + 1/2)**(p+1) - (i1 - 1/2)**(p+1))
                                )
                        elif ((i1 - 1/2) < np.sqrt(r**2 - (j1 - 1/2)**2)
                                < (i1 + 1/2) and (i1 - 1/2) <
                                np.sqrt(r**2 - (j1 + 1/2)**2)
                                < (i1 + 1/2)):
                            m_pq[r_ind - j1, r_ind + i1] = (
                                (1 / (q+1)) *
                                integral_I(np.sqrt(r**2 - (j1 + 1/2)**2),
                                           np.sqrt(r**2 - (j1 - 1/2)**2)) +
                                (1 / ((p+1)*(q+1))) *
                                ((j1 + 1/2)**(q+1) *
                                 ((r**2 - (j1 + 1/2)**2)**((p+1)/2) -
                                 (i1 - 1/2)**(p+1)) - (j1 - 1/2)**(q+1) *
                                 ((r**2 - (j1 - 1/2)**2)**((p+1)/2) -
                                  (i1 - 1/2)**(p+1)))
                                )
                        elif ((i1 - 1/2) < np.sqrt(r**2 - (j1 + 1/2)**2)
                                < (i1 + 1/2) and (j1 - 1/2) <
                                np.sqrt(r**2 - (i1 + 1/2)**2)
                                < (j1 + 1/2)):
                            m_pq[r_ind - j1, r_ind + i1] = (
                                (1 / (q+1)) *
                                integral_I(np.sqrt(r**2 - (j1 + 1/2)**2),
                                           (i1 + 1/2)) +
                                (1 / ((p+1)*(q+1))) *
                                ((j1 + 1/2)**(q+1) *
                                 ((r**2 - (j1 + 1/2)**2)**((p+1)/2) -
                                 (i1 - 1/2)**(p+1)) - (j1 - 1/2)**(q+1) *
                                 ((i1 + 1/2)**(p+1) - (i1 - 1/2)**(p+1)))
                                )
                m_pq[r_ind - j1, r_ind - i1] = (
                        (-1)**p * m_pq[r_ind - j1, r_ind + i1])
                m_pq[r_ind + j1, r_ind + i1] = (
                        (-1)**q * m_pq[r_ind - j1, r_ind + i1])
                m_pq[r_ind + j1, r_ind - i1] = (
                        (-1)**(p+q) * m_pq[r_ind - j1, r_ind + i1])
        return m_pq

    def compute(self, img):
        m00 = np.sum(img * self.mask00)
        m10 = np.sum(img * self.mask10)
        m01 = np.sum(img * self.mask01)
        m11 = np.sum(img * self.mask11)
        m20 = np.sum(img * self.mask20)
        m02 = np.sum(img * self.mask02)
        return m00, m10, m01, m11, m20, m02


class SubpixelCornerEdgeDetector:
    """
    Class that implements the corner and edge detection on the image
    """

    def __init__(self, img, pixel_coords, d=15):

        assert (d % 2 != 0) and isinstance(d, int), "'d' must be odd integer"
        self.d = d
        self.img = img_as_float(rgb2gray(img))

        pixel_coords_ = list()
        pixel_coords = np.array(pixel_coords)
        for k in range(pixel_coords.shape[0]):
            if (self.d < pixel_coords[k, 0] < self.img.shape[0] - self.d
                    and self.d < pixel_coords[k, 1] < self.img.shape[1] - self.d):
                pixel_coords_.append(pixel_coords[k])
        pixel_coords_ = np.array(pixel_coords_)
        self.pixel_coords = pixel_coords_

        self.img_moments = CircImgMoments(d)

        self.p_img = np.full(self.img.shape, np.nan)
        self.theta_img = np.full(self.img.shape, np.nan)
        self.t_img = np.full(self.img.shape, np.nan)
        self.a_img = np.full(self.img.shape, np.nan)
        self.b_img = np.full(self.img.shape, np.nan)
        self.phi_img = np.full(self.img.shape, np.nan)

        self.Ms_img = np.full(self.img.shape, np.nan)
        self.Ml_img = np.full(self.img.shape, np.nan)
        self.Me_img = np.full(self.img.shape, np.nan)
        self.Mc_hat_img = np.full(self.img.shape, np.nan)
        self.Me_hat_img = np.full(self.img.shape, np.nan)
        self.x_p_img = np.full(self.img.shape, np.nan)
        self.y_p_img = np.full(self.img.shape, np.nan)

        self.corners_map = np.full(self.img.shape, np.nan)
        self.edges_map = np.full(self.img.shape, np.nan)
        self.corner_subpixel_coords = list()
        self.edge_subpixel_coords = list()


    def __calculate_theta(self, m10, m01):
        if (m10 == 0) and (m01 == 0):
            theta = 0
        elif (m10 == 0) and (m01 != 0):
            theta = np.sign(m01)*np.pi/2
        elif (m10 < 0) and (m01 > 0):
            theta = np.pi + np.arctan(m01/m10)
        elif (m10 < 0) and (m01 < 0):
            theta = - np.pi + np.arctan(m01/m10)
        else:
            theta = np.arctan(m01/m10)
        return theta

    def __compute_canonical_moments(self, m00, m10, m01, m11, m20, m02, h_c):
        M00 = m00
        M10 = np.sqrt(m01**2 + m10**2)
        if (M00 != 0) and (M10/M00 > h_c):
            M20 = (m10**2 * m20 + 2 * m01 * m10 * m11
                   + m01**2 * m02)/(m01**2 + m10**2)
            M02 = (m10**2 * m02 - 2 * m01 * m10 * m11
                   + m01**2 * m20)/(m01**2 + m10**2)
            return M00, M10, M20, M02
        else:
            return None

    def __calculate_t(self, J, M10, p, r):
        t = None

        eq = (J**2 * p**2 + 2*J*p*(5*r**2 - 4*p**2)*M10
              + r**2 * (9*r**2 - 8*p**2) * M10**2)
        if p == 0:
            t = -2/3 * J/M10
        elif (p != 0) and (eq >= 0):
            t1 = ((-J*p + 3*M10*r**2 + np.sqrt(eq)) / (4*M10*p))
            t2 = ((-J*p + 3*M10*r**2 - np.sqrt(eq)) / (4*M10*p))
            if abs(t1) < r:
                t = t1
            elif abs(t2) < r:
                t = t2
        return t

    def __calculate_a_b(self, M00, M10, M20, M02, p, r, t):
        a = None
        b = None

        if p == 0 and t == 0:
            a = M00/(np.pi * r**2) - (3 * M10)/(4*r**3)
            b = M00/(np.pi * r**2) + (3 * M10)/(4*r**3)
        else:
            eq_sqrt = (np.sqrt(r**2-t**2) *
                       (2*p*t**2 - r**2 * p - 3*r**2*t + t*p**2 + p**3))
            if eq_sqrt != 0:
                L = ((1/(np.pi * r**2)) *
                     (M00 - ((6*(M20-M02))/eq_sqrt) *
                     (p*np.sqrt(r**2-t**2) + r**2*np.arcsin(t/r))))
                if (p-t) <= 0:
                    a = (L-(3/2*M10) / (np.sqrt(r**2-t**2)*(2*r**2-p*t-p**2)))
                    b = (L+(3/2*M10) / (np.sqrt(r**2-t**2)*(2*r**2-p*t-p**2)))
                elif (p-t) > 0:
                    a = (L+(3/2*M10) / (np.sqrt(r**2-t**2)*(2*r**2-p*t-p**2)))
                    b = (L-(3/2*M10) / (np.sqrt(r**2-t**2)*(2*r**2-p*t-p**2)))
        return a, b

    def __calculate_phi(self, p, r, t):
        if t == p:
            phi = np.pi/2
        else:
            phi = np.arctan(np.sqrt(r**2-t**2) / abs(t-p))
        return phi

    def __calculate_Ms(self, img, theta, r, a, b):
        img = rotate(img, np.rad2deg(-theta), order=1)
        img_reflect = img[::-1, :]
        img = img * self.img_moments.mask00
        img_reflect = img_reflect * self.img_moments.mask00
        Ms = 1 - np.sum(np.abs(img - img_reflect))/(np.pi * r**2 * np.abs(a-b))
        return Ms

    def __calculate_Ml(self, H, J, M10, r):
        w_1 = H/(3*M10)
        w_2 = -J/M10
        w_3 = (H-J)/(4*M10)

        Ml = np.std([w_1, w_2, w_3])/r
        return Ml

    def __calculate_Me(self, H, J, M10, r):
        v_1 = 2/3 * H/M10
        v_2 = -2/3 * J/M10
        v_3 = (H-J)/(3*M10)

        Me = 1 - np.std([v_1, v_2, v_3])/r
        return Me

    def __algorithm_1(self, x, y, h_c, h_ab, h_s):
        r = self.d / 2
        r_ind = int(np.fix(r))

        moments = self.img_moments.compute(self.img[x - r_ind:x + r_ind + 1,
                                                    y - r_ind:y + r_ind + 1])
        (m00, m10, m01, m11, m20, m02) = moments
        theta = self.__calculate_theta(m10, m01)

        canonical_moments = self.__compute_canonical_moments(
                m00, m10, m01, m11, m20, m02, h_c)
        if canonical_moments:
            (M00, M10, M20, M02) = canonical_moments

            H = 4 * M20 - M00 * r ** 2
            J = 4 * M02 - M00 * r ** 2
            p = ((H + J) / (2 * M10))

            if abs(p) < r:
                t = self.__calculate_t(J, M10, p, r)

                if t:
                    phi = self.__calculate_phi(p, r, t)
                    (a, b) = self.__calculate_a_b(M00, M10, M20, M02, p, r, t)

                    if a and b:
                        if abs(a - b) > h_ab:
                            Ms = self.__calculate_Ms(
                                    self.img[x - r_ind:x + r_ind + 1,
                                             y - r_ind:y + r_ind + 1],
                                    theta, r, a, b)

                            if Ms > h_s:
                                self.Ms_img[x, y] = Ms
                                cos_theta = m10 / M10
                                sin_theta = m01 / M10
                                self.x_p_img[x, y] = x - p * sin_theta
                                self.y_p_img[x, y] = y + p * cos_theta

                                Ml = self.__calculate_Ml(H, J, M10, r)
                                Me = self.__calculate_Me(H, J, M10, r)
                                self.Ml_img[x, y] = Ml
                                self.Me_img[x, y] = Me
                                self.Mc_hat_img[x, y] = Ml * Me
                                self.Me_hat_img[x, y] = (1-Ml) * Me

                                self.p_img[x, y] = p

                                self.theta_img[x, y] = theta
                                self.t_img[x, y] = t
                                self.a_img[x, y] = a
                                self.b_img[x, y] = b
                                self.phi_img[x, y] = phi
                                return 1
        return 0

    def __algorithm_2(self, h_c_hat, min_dist):
        Mc_hat_img = self.Mc_hat_img.copy()

        Mc_hat_img[np.isnan(Mc_hat_img)] = 0
        Mc_hat_img[Mc_hat_img <= h_c_hat] = 0

        corners_map = corner_peaks(Mc_hat_img,
                                   min_distance=int(min_dist),
                                   threshold_abs=None,
                                   threshold_rel=None,
                                   exclude_border=True,
                                   indices=False)
        return corners_map

    def __algorithm_3(self, h_e_hat, min_dist):
        Me_hat_img = self.Me_hat_img.copy()

        Me_hat_img[np.isnan(Me_hat_img)] = 0
        Me_hat_img[Me_hat_img <= h_e_hat] = 0

        edges_map = corner_peaks(Me_hat_img,
                                 min_distance=int(min_dist),
                                 threshold_abs=None,
                                 threshold_rel=None,
                                 exclude_border=True,
                                 indices=False)
        return edges_map

    def __get_subpixel_coords(self, map):
        inds = np.nonzero(map)
        x_sub = self.x_p_img[inds]
        y_sub = self.y_p_img[inds]
        subpixel_coords = np.vstack((x_sub, y_sub)).T
        return subpixel_coords.tolist()

    def detect_corners(self, h_ab, h_c, h_s, h_c_hat, min_dist=1):
        for k in range(self.pixel_coords.shape[0]):
            x = self.pixel_coords[k, 0]
            y = self.pixel_coords[k, 1]
            self.__algorithm_1(x, y, h_c, h_ab, h_s)

        self.corners_map = self.__algorithm_2(h_c_hat, min_dist)

        self.corner_subpixel_coords = self.__get_subpixel_coords(self.corners_map)
        return self.corner_subpixel_coords

    def detect_edges(self, h_ab, h_c, h_s, h_e_hat, min_dist=1):
        for k in range(self.pixel_coords.shape[0]):
            x = self.pixel_coords[k, 0]
            y = self.pixel_coords[k, 1]
            self.__algorithm_1(x, y, h_c, h_ab, h_s)

        self.edges_map = self.__algorithm_3(h_e_hat, min_dist)

        self.edge_subpixel_coords = self.__get_subpixel_coords(self.edges_map)
        return self.edge_subpixel_coords
