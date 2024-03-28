import cv2
import matplotlib.pyplot as plt
from skimage import draw
import numpy as np
from scipy.interpolate import interp1d
from torch import Tensor

def find_contours(mask: Tensor):
    h,w = mask.shape
    if isinstance(mask, Tensor): 
        mask = mask.numpy().astype(np.uint8)
    edge = np.zeros((h,w), dtype=np.uint8)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    assert len(contours) == 1
    contours = contours[0].squeeze()
    edge[contours[:,1], contours[:,0]] = 1

    # check
    assert edge.sum() == (edge*mask).sum()
    return edge

def find_contour_points(mask: Tensor):
    '''
        mask: (h,w), 0 or 1
        return: contours (n,2)
                the x,y of the points 
    '''
    h,w = mask.shape
    mask = mask.numpy().astype(np.uint8)
    edge = np.zeros((h,w), dtype=np.uint8)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # assert len(contours) == 1
    if len(contours) != 1:
        return np.array([])
    
    contours = contours[0].squeeze()
    edge[contours[:,1], contours[:,0]] = 1

    # check
    assert edge.sum() == (edge*mask).sum()
    return contours

def interpolate_contour(contour, num_pts, start_pt=None, end_pt=None):
    """ interpolate a contour to the desired number of points """
    x_pts = contour[:, 0]
    y_pts = contour[:, 1]
    total_pts = x_pts.shape[0]
    if start_pt is not None:
        assert end_pt is not None, "end_pt must be provided if start_pt is"
        spaced_x, spaced_y = interpolate_regularly_spaced_contour(contour, num_pts, closed=False)
        x = np.hstack([start_pt[0], spaced_x[1:-1], end_pt[0]]).ravel()
        y = np.hstack([start_pt[1], spaced_y[1:-1], end_pt[1]]).ravel()
        return x, y
    else:
        return interpolate_regularly_spaced_contour(contour, num_pts, closed=False)

def interpolate_regularly_spaced_contour(contour, num_pts, closed=True):
    """ interpolate a contour to the desired number of points where all points are evenly spaced a long the contour """
    if closed:
        contour = contour.copy()
        contour = np.vstack([contour, contour[0]])
    x_pts = contour[:, 0]
    y_pts = contour[:, 1]
    total_pts = x_pts.shape[0]
    dists = [np.linalg.norm(contour[i + 1] - contour[i]) for i in range(total_pts - 1)]
    inds = [0] + [sum(dists[:i]) / sum(dists) for i in range(1, total_pts)]
    inds[-1] = 1
    inds = np.array(inds)
    # having the same point twice will cause interp to fail
    zero_dist_mask = np.ones(total_pts, dtype=bool)
    zero_dist_mask[np.where(np.array(dists) < 1e-8)[0]] = False
    inds = inds[zero_dist_mask]
    x_pts = x_pts[zero_dist_mask]
    y_pts = y_pts[zero_dist_mask]
    x_interp = interp1d(inds, x_pts, kind='cubic')
    y_interp = interp1d(inds, y_pts, kind='cubic')
    index_select = np.linspace(0, 1., num_pts)
    x = x_interp(index_select)
    y = y_interp(index_select)
    return x, y

def trace_to_mask(t, mask_size: tuple):
    """ creates a 2D segmentation mask from a keypoint vector N*2 with size N and mask_size is the size of the resulting mask image """
    x = t[:, 0]*mask_size[0]
    y = t[:, 1]*mask_size[1]

    r, c = draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), (mask_size))
    mask = np.zeros(mask_size, np.uint8)
    mask[r, c] = 1
    return mask

def echonet_trace_to_mask(t, mask_size: tuple):
    """ creates a 2D segmentation mask from a echonet-specific keypoint vector N*4 with size 2N and mask_size = resulting image size
    The keypoint vector consists of parallel point pairs (x1,y1) and (x2,y2) so that the second pair needs to be flipped
    to get a valid contour
    """
    x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))

    r, c = draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), (mask_size))
    mask = np.zeros(mask_size, np.uint8)
    mask[r, c] = 1
    return mask

class GradientCurvature:

    def __init__(self, trace, plot_derivatives=False):
        self.trace = trace
        self.plot_derivatives = plot_derivatives
        self.curvature = None

    def _get_gradients(self):
        self.x_trace = [x[0] for x in self.trace]
        self.y_trace = [y[1] for y in self.trace]

        x_prime = np.gradient(self.x_trace)
        y_prime = np.gradient(self.y_trace)
        x_bis = np.gradient(x_prime)
        y_bis = np.gradient(y_prime)

        if self.plot_derivatives:
            plt.subplot(211)
            plt.plot(x_prime)
            plt.plot(y_prime)
            plt.subplot(212)
            plt.plot(x_bis)
            plt.plot(y_bis)

        return x_prime, y_prime, x_bis, y_bis

    def calculate_curvature(self):
        x_prime, y_prime, x_bis, y_bis = self._get_gradients()
        curvature = x_prime * y_bis / ((x_prime ** 2 + y_prime ** 2) ** (3/2)) - \
            y_prime * x_bis / ((x_prime ** 2 + y_prime ** 2) ** (3/2))  # Numerical trick to get accurate values
        self.curvature = curvature
        return curvature

class LeftVentricleUnorderedContour():
    """ class to handle an unordered LV contour. Can be initialized from a binary mask or a point set"""
    apex_version = "basal_points"
    points_to_use_in_internal_storage = 80

    def __init__(self, mask=None, contour=None):
        """ Contour should be shape Nx2 and in X, Y format """
        if contour is not None:
            self.contour = contour
        elif mask is not None:
            self.contour = self._mask_to_contour(mask)
        else:
            raise ValueError("either mask or contour must be defined")
        assert self.contour.shape[1] == 2, f"contour should be shape Nx2, found {contour.shape}"
        assert len(self.contour.shape) == 2, f"contour should be shape Nx2, found {contour.shape}"
        # self.contour = self._check_direction(self.contour)
        # self.contour = self._roll_to_apex(self.contour)  # this will ensure basal_points method works properly
        self.contour = np.vstack(interpolate_regularly_spaced_contour(self.contour, self.points_to_use_in_internal_storage)).T

        # landmarks
        # BASAL POINTS
        self.basal_left, self.basal_right = self._get_basal_pts_from_contour()
        # APEX: rough estimation for apex location is the point farthest away from the two basal points. In the case that the basal_points
        # are not known we can just approximate this as the point lowest point (highest in an image)
        if self.apex_version == "basal_points":
            self.apex = self._apex_found_from_farthest_from_basal_points()
        else:
            self.apex = self._apex_found_from_min_pt()

    @property
    def x(self):
        return self.contour[:, 0]

    @property
    def y(self):
        return self.contour[:, 1]

    def _roll_to_apex(self, contour):
        apex = self._apex_found_from_min_pt()
        apex_ind = np.argmin(np.linalg.norm(apex - self.contour, axis=1))
        return np.roll(contour, apex_ind, axis=0)

    @staticmethod
    def _check_direction(contour):
        """ ensure contour moves from apex -> basal_right -> basal_left. Assumes apex is already at point 0 """
        if contour[5, 0] < contour[0, 0]:
            contour = contour[::-1]
        return contour

    @staticmethod
    def _mask_to_contour(mask):
        def smooth_contour(img, k=None, s=3):
            if k is None:
                k = int(np.sqrt(img.shape[0]))
                if k % 2 == 0:
                    k -= 1
            for i in range(s):
                img = cv2.GaussianBlur(img, (k, k), 0)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
            return img

        def get_largest_contour(contours):
            """ find all contours above threshold """
            largest = None
            current_biggest = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > current_biggest:
                    largest = contour
                    current_biggest = area
            if largest is None:
                raise ValueError("no contours in image > 0 area")
            return largest

        mask = smooth_contour(mask)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour = get_largest_contour(contours).squeeze()
        # contour = contour[::5]  # too many points for accurate angle detection
        return contour

    def _apex_found_from_min_pt(self):
        return np.array([np.min(self.x), self.y[np.argmin(self.x)]])

    def _apex_found_from_farthest_from_basal_points(self):
        basal_left, basal_right = self._get_basal_pts_from_contour()
        dists = np.linalg.norm(self.contour - basal_left, axis=1) + np.linalg.norm(self.contour - basal_right, axis=1)
        return self.contour[np.argmax(dists)]

    def _get_basal_pts_from_contour(self, plot=False):
        """ estimate basal points, find by high angles """
        def plot_find_basal_points():
            plt.figure(figsize=(7, 3))
            plt.plot(curvature, label="curvature")
            plt.plot(mask, label="mask")
            plt.plot(pt0_mask, label="pt0_maks")
            plt.plot(basal_pt0, curvature[basal_pt0], "ro", label="basel_pt0")
            plt.plot(basal_pt1, curvature[basal_pt1], "go", label="basel_pt1")
            plt.legend()

        c = self.contour[::3]

        vecs = c[1:] - c[:-1]  # vectors from one point to another
        # angles = [distance.cosine(l, r) for l, r in zip(vecs[:-1], vecs[1:])]  # angles from one point to another
        gc = GradientCurvature(c)
        curvature = abs(gc.calculate_curvature())

        # it is possible to have high angles around the apex so we should mask out any points that could be close
        cutoff = np.min(c[:, 1]) + .6 * (np.max(c[:, 1]) - np.min(c[:, 1]))
        mask = ~(c[:, 1] > cutoff)  # [1:-1]  # 1:-1 because we lose two points, 1 each during vec and angle creation
        angles_masked = np.ma.masked_array(curvature, mask=mask)

        basal_pt0 = angles_masked.argsort(endwith=False)[-1] #+ 1  # +1 from the one point we lost at the beginning
        # mask out points around this point from consideration of other point
        pt0_mask = np.zeros(len(curvature), dtype=bool)
        pt0_mask[int(basal_pt0-.1*len(curvature)):int(basal_pt0+.1*len(curvature))] = True
        angles_masked = np.ma.masked_array(curvature, mask=np.ma.mask_or(mask, pt0_mask))
        basal_pt1 = angles_masked.argsort(endwith=False)[-1] #+ 1

        if plot:
            plot_find_basal_points()

        if abs(basal_pt1 - basal_pt0) > .5 * c.shape[0]:
            print(f"WARNING: basal points are quite far apart, possible error: {basal_pt0}, {basal_pt1}")
            plot_find_basal_points()

        # figure out which one is which
        if c[basal_pt0, 0] < c[basal_pt1, 0]:
            basal_left, basal_right = basal_pt0, basal_pt1
        else:
            basal_right, basal_left = basal_pt0, basal_pt1
        return c[basal_left], c[basal_right]

    def plot(self, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.plot(self.x, self.y, 'b-')
        ax.plot(self.x[1], self.y[1], "ko")
        ax.plot(*self.apex, "bo", label="apex")
        ax.plot(*self.basal_left, "ro", label="basal_left")
        ax.plot(*self.basal_right, "go", label="basal_right")
        ax.legend()

    def to_ordered_contour(self, num_pts=12, parts=("myo",)):
        ordered_contour = dict()
        for part in parts:
            ordered_contour[part] = self._get_orderd_contour_part(self.contour, self.basal_left, self.basal_right,
                                                                  self.apex, num_pts, part)
        return ordered_contour

    @staticmethod
    def _get_orderd_contour_part(contour, basal_left, basal_right, apex, num_pts=13, part="myo"):
        """
        Extract a part of the LV contour with the given number of points (num_pts) and section (part)

        Args:
            contour: a numpy array containing the contour
            basal_left: estimation of left basal point
            basal_right: estimation of right basal point
            apex: estimation of apex
            num_pts: how many points to include in the contour
            part:  either "myo" (just the LV contour), "mv" (just the mv contour), or "closed" (everything).
              "closed" calls this function twice, one with "myo" and one with "mv" to ensure that the basal points are
              always the same index (very important for graph convolutional networks).

        Returns: arrays for x, and y points with num_pts points

        """
        basal_left_ind = np.argmin(np.linalg.norm(basal_left - contour, axis=1))
        # start with basal left
        contour = np.roll(contour, -basal_left_ind, axis=0)
        # now find basal right
        basal_right_ind = np.argmin(np.linalg.norm(basal_right - contour, axis=1))
        # make sure we move the correct way around the contour (for now just find the long way)
        if basal_right_ind < contour.shape[0] - basal_right_ind:
            contour[1:] = contour[1:][::-1]
            basal_right_ind = np.argmin(np.linalg.norm(basal_right - contour, axis=1))
        apex_ind = np.argmin(np.linalg.norm(apex - contour, axis=1))
        # cut contour at other basal point depending on desired region
        if part == "myo":
            x_l, y_l = interpolate_contour(contour[:apex_ind], num_pts // 2, basal_left, apex)
            x_r, y_r = interpolate_contour(contour[apex_ind:basal_right_ind], num_pts // 2 + 1, apex, basal_right)
            x = np.hstack([x_l, x_r[1:]]).ravel()
            y = np.hstack([y_l, y_r[1:]]).ravel()
        elif part == "mv":
            x, y = interpolate_contour(contour[basal_right_ind:], num_pts, basal_right, basal_left)
        elif part == "closed":
            myo_num_pts = int(0.9 * num_pts)
            mv_num_pts = num_pts - myo_num_pts + 1
            assert myo_num_pts > 5 and mv_num_pts > 1, "need more pts for a closed contour"
            m_x, m_y = interpolate_contour(contour[:basal_right_ind], myo_num_pts, basal_left, basal_right)
            v_x, v_y = interpolate_contour(contour[basal_right_ind:], mv_num_pts, basal_right, basal_left)
            x = np.hstack([m_x, v_x[1:]]).ravel()
            y = np.hstack([m_y, v_y[1:]]).ravel()
        else:
            raise ValueError(f"part {part} not recognized")
        return x, y
