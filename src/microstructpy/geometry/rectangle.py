"""Module for Rectangle class."""
# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #
from __future__ import division

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from microstructpy.geometry.n_box import NBox

__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# Rectangle Class                                                             #
#                                                                             #
# --------------------------------------------------------------------------- #
class Rectangle(NBox):
    """Rectangle

    This class contains a generic, 2D rectangle. The position and dimensions
    of the box can be specified using any of the parameters below.

    Without parameters, this returns a unit square centered on the origin.

    Args:
        length (float): *(optional)* Length of the rectangle.
        width (float): *(optional)* Width of the rectangle. *(optional)*
        side_lengths (list): *(optional)* Side lengths. Defaults to (1, 1).
        center (list): *(optional)* Center of rectangle. Defaults to (0, 0).
        corner (list): *(optional)* Bottom-left corner.
        bounds (list): *(optional)* Bounds of rectangle. Expected to be in the
            format [(xmin, xmax), (ymin, ymax)].
        limits : Alias for *bounds*.
        angle (float): *(optional)* The rotation angle, in degrees.
        angle_deg (float): *(optional)* The rotation angle, in degrees.
        angle_rad (float): *(optional)* The rotation angle, in radians.
        matrix (numpy.ndarray): *(optional)* The 2x2 rotation matrix.
    """

    def __init__(self, **kwargs):
        if 'length' in kwargs and 'width' in kwargs:
            kwargs['side_lengths'] = [kwargs['length'], kwargs['width']]

        if 'angle' in kwargs:
            cp = np.cos(np.radians(kwargs['angle']))
            sp = np.sin(np.radians(kwargs['angle']))
            kwargs['matrix'] = np.array([[cp, -sp], [sp, cp]])
        elif 'angle_deg' in kwargs:
            cp = np.cos(np.radians(kwargs['angle_deg']))
            sp = np.sin(np.radians(kwargs['angle_deg']))
            kwargs['matrix'] = np.array([[cp, -sp], [sp, cp]])
        elif 'angle_rad' in kwargs:
            cp = np.cos(kwargs['angle_rad'])
            sp = np.sin(kwargs['angle_rad'])
            kwargs['matrix'] = np.array([[cp, -sp], [sp, cp]])

        NBox.__init__(self, **kwargs)
        try:
            self.center
        except AttributeError:
            self.center = [0, 0]

        try:
            self.side_lengths
        except AttributeError:
            self.side_lengths = [1, 1]

    # ----------------------------------------------------------------------- #
    # Best Fit Function                                                       #
    # ----------------------------------------------------------------------- #
    def best_fit(self, points):
        """Find rectangle of best fit for points

        Args:
            points (list): List of points to fit.

        Returns:
            Rectangle: an instance of the class that best fits the points.
        """
        # Unpack the input points
        pts = np.array(points, dtype='float')
        x, y = pts.T  # pylint: disable=E0633
        # E0633: unpacking-non-sequence

        # Find the most likely orientation for the rectangle
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        theta = np.arctan(m)

        # Rotate the points to an axis-aligned frame
        s = np.sin(theta)
        c = np.cos(theta)
        rot = np.array([[c, -s], [s, c]])
        pts_prin = pts.dot(rot)

        # Translate points to center of bounding box
        mins = pts_prin.min(axis=0)
        maxs = pts_prin.max(axis=0)
        bb_cen = 0.5 * (mins + maxs)
        pts_bb = pts_prin - bb_cen
        rect_fit = _axis_aligned_best_fit(pts_bb)

        # Translate center to rotated frame
        fit_cen_prin = rect_fit.center + bb_cen
        rect_fit.center = rot.dot(fit_cen_prin.reshape(-1, 1)).flatten()

        # Disambiguate the orientation and axes
        x_ax_seed = np.array(self.matrix)[:, 0]
        x_dot, y_dot = rot.T.dot(x_ax_seed)

        if np.abs(x_dot) > np.abs(y_dot):
            x_ax_fit = np.sign(x_dot) * rot[:, 0]
        else:
            x_ax_fit = np.sign(y_dot) * rot[:, 1]
            rect_fit.side_lengths = [rect_fit.width, rect_fit.length]

        ang_diff = np.arcsin(np.cross(x_ax_seed, x_ax_fit))
        ang_rad = self.angle_rad + ang_diff
        rect_fit.angle_rad = ang_rad

        return rect_fit

    # ----------------------------------------------------------------------- #
    # Representation Function                                                 #
    # ----------------------------------------------------------------------- #
    def __repr__(self):
        repr_str = 'Rectangle('
        repr_str += 'center=' + repr(tuple(self.center)) + ', '
        repr_str += 'side_lengths=' + repr(tuple(self.side_lengths)) + ', '
        repr_str += 'angle=' + repr(self.angle) + ')'
        return repr_str

    # ----------------------------------------------------------------------- #
    # Number of Dimensions                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def n_dim(self):
        """int: Number of dimensions, 2"""
        return 2

    # ----------------------------------------------------------------------- #
    # Area                                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def area(self):
        """float: Area of rectangle"""
        return self.n_vol

    @classmethod
    def area_expectation(cls, **kwargs):
        r"""Expected area of rectangle

        This method computes the expected area of a rectangle. There are two
        main ways to define the size of a rectangle: by the length and width
        and by the bounds. If the input rectangle is defined by length and
        width, the expected area is:

        .. math::

            \mathbb{E}[A] = \mathbb{E}[L W] = \mu_L \mu_W

        For the case where it is defined by upper and lower bounds:

        .. math::

            \mathbb{E}[A] = \mathbb{E}[(X_U - X_L) (Y_U - Y_L)]

        .. math::
            \mathbb{E}[A] =
            \mu_{X_U}\mu_{Y_U} - \mu_{X_U} \mu_{Y_L} -
            \mu_{X_L}\mu_{Y_U} + \mu_{X_L}\mu_{Y_L}

        Example:
            >>> import scipy.stats
            >>> import microstructpy as msp
            >>> L = scipy.stats.uniform(loc=1, scale=2)
            >>> W = scipy.stats.norm(loc=3.2, scale=5.1)
            >>> L.mean() * W.mean()
            6.4
            >>> msp.geometry.Rectangle.area_expectation(length=L, width=W)
            6.4

        Args:
            **kwargs: Keyword arguments, same as :class:`.Rectangle` but the
                inputs can be from the :mod:`scipy.stats` module.

        Returns:
            float: Expected/average area of rectangle.

        """
        if 'length' in kwargs or 'width' in kwargs:
            len_dist = kwargs.get('length', 1)
            width_dist = kwargs.get('width', 1)
            return _prod_exp(*[len_dist, width_dist])

        if 'side_lengths' in kwargs:
            return _prod_exp(*kwargs['side_lengths'])

        if 'bounds' in kwargs:
            x_bnds, y_bnds = kwargs['bounds']
            x_lb, x_ub = x_bnds
            y_lb, y_ub = y_bnds

            p1 = _prod_exp(*[x_ub, y_ub])
            p2 = _prod_exp(*[x_ub, y_lb])
            p3 = _prod_exp(*[x_lb, y_ub])
            p4 = _prod_exp(*[x_lb, y_lb])

            return p1 - p2 - p3 + p4

        raise ValueError('Cannot find the expected area of rectangle')

    # ----------------------------------------------------------------------- #
    # Properties                                                              #
    # ----------------------------------------------------------------------- #
    @property
    def length(self):
        """float: Length of rectangle, side length along 1st axis"""
        return self.side_lengths[0]

    @property
    def width(self):
        """float: Width of rectangle, side length along 2nd axis"""
        return self.side_lengths[1]

    @property
    def angle(self):
        """float: Rotation angle of rectangle - degrees"""
        return self.angle_deg

    @property
    def angle_deg(self):
        """float: Rotation angle of rectangle - degrees"""
        return 180 * self.angle_rad / np.pi

    @property
    def angle_rad(self):
        """float: Rotation angle of rectangle - radians"""
        return np.arctan2(self.matrix[1][0], self.matrix[0][0])

    @angle_rad.setter
    def angle_rad(self, t):
        cp = np.cos(t)
        sp = np.sin(t)
        self.matrix = np.array([[cp, -sp], [sp, cp]])

    # ----------------------------------------------------------------------- #
    # Circle Approximation                                                    #
    # ----------------------------------------------------------------------- #
    def approximate(self, x1=None):
        """Approximate rectangle with a set of circles.

        This method approximates a rectangle with a set of circles.
        These circles are spaced uniformly along the long axis of the
        rectangle with distance ``x1`` between them.

        Example
        -------

        For a rectangle with length=2.5, width=1, and x1=0.3,
        the approximation would look like :numref:`f_api_rectangle_approx`.

        .. _f_api_rectangle_approx:
        .. figure:: ../../auto_examples/geometry/images/sphx_glr_plot_rectangle_001.png

            Circular approximation of rectangle.

        Args:
            x1 (float or None): *(optional)* Spacing between the circles.
                If not specified, the spacing is 0.25x the length of the
                shortest side.

        Returns:
            numpy.ndarray: An Nx3 array, where each row is a circle and the
            columns are x, y, and r.

        """  # NOQA: E501
        if x1 is None:
            x1 = 0.25 * min(self.side_lengths)

        if self.side_lengths[0] >= self.side_lengths[1]:
            length = self.side_lengths[0]
            width = self.side_lengths[1]
            inds = [0, 1]
        else:
            length = self.side_lengths[1]
            width = self.side_lengths[0]
            inds = [1, 0]

        # Centerline circles
        xc = 0
        half_len = 0.5 * length
        r = 0.5 * width
        circs = []
        while xc < half_len:
            circ = [xc, 0, r]
            circs.append(circ)

            if np.isclose(xc + r, half_len):
                xc = half_len
            elif xc + x1 > half_len - r:
                xc = half_len - r
            else:
                xc = xc + x1

        # Corner circle
        while r > 0:
            x_init = circs[-1][0]
            y_init = circs[-1][1]
            x = x_init + x1
            y = y_init + x1

            dx = half_len - x
            dy = 0.5 * width - y
            r = min(dx, dy)
            if r <= 0:
                break
            circs.append([x, y, r])

        # Reflect circles
        circs = np.array(circs)
        for dim in range(self.n_dim):
            mask = circs[:, dim] > 0
            new_circs = np.copy(circs[mask])
            new_circs[:, dim] *= -1
            circs = np.concatenate((circs, new_circs))

        circs[:, inds] = circs[:, [0, 1]]

        # Rotate and translate circles
        pts = circs[:, :-1]
        circs[:, :-1] = pts.dot(np.array(self.matrix).T) + self.center
        return circs

    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    def plot(self, **kwargs):
        """Plot the rectangle.

        This function adds a :class:`matplotlib.patches.Rectangle` patch to the
        current axes. The keyword arguments are passed through to the patch.

        Args:
            **kwargs (dict): Keyword arguments for the patch.

        """  # NOQA: E501
        pt = self.corner
        w = self.length
        h = self.width
        ang = self.angle
        c = patches.Rectangle(pt, w, h, ang, **kwargs)
        plt.gca().add_patch(c)


def _prod_exp(*args):
    prod = 1
    for arg in args:
        try:
            arg_mu = arg.moment(1)
        except AttributeError:
            arg_mu = arg
        prod *= arg_mu
    return prod

def _axis_aligned_best_fit(pts):
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # Find nearest edge
    x1_dist = np.abs(pts[:, 0] - x_min)
    x2_dist = np.abs(pts[:, 0] - x_max)
    y1_dist = np.abs(pts[:, 1] - y_min)
    y2_dist = np.abs(pts[:, 1] - y_max)

    dists = np.array([x1_dist, x2_dist, y1_dist, y2_dist]).T
    min_dist = dists.min(axis=1)

    masks = [np.isclose(d, min_dist) for d in dists.T]
    fits = [np.mean(pts[m, i //  2].T) for i, m in enumerate(masks)]

    xc = 0.5 * (fits[0] + fits[1])
    yc = 0.5 * (fits[2] + fits[3])
    x_len = fits[1] - fits[0]
    y_len = fits[3] - fits[2]

    return Rectangle(center=[xc, yc], length=x_len, width=y_len)


# --------------------------------------------------------------------------- #
#                                                                             #
# Square Class                                                                #
#                                                                             #
# --------------------------------------------------------------------------- #
class Square(Rectangle):
    """A square.

    This class contains a generic, 2D square. It is derived from the
    :class:`microstructpy.geometry.Rectangle` class and contains the
    ``side_length`` property, rather than multiple side lengths.

    Args:
        side_length (float): *(optional)* Side length. Defaults to 1.
        center (list): *(optional)* Center of rectangle. Defaults to (0, 0).
        corner (list): *(optional)* Bottom-left corner.
    """

    def __init__(self, **kwargs):
        if 'side_length' in kwargs:
            kwargs['side_lengths'] = 2 * [kwargs['side_length']]

        Rectangle.__init__(self, **kwargs)

    # ----------------------------------------------------------------------- #
    # Side Length Property                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def side_length(self):
        """float: length of the side of the square."""
        return self.side_lengths[0]

    # ----------------------------------------------------------------------- #
    # Area                                                                    #
    # ----------------------------------------------------------------------- #
    @classmethod
    def area_expectation(cls, **kwargs):
        r"""Expected area of square

        This method computes the expected area of a square with distributed
        side length.
        The expectation is:

        .. math::

            \mathbb{E}[A] = \mathbb{E}[S^2] = \mu_S^2 + \sigma_S^2

        Example:
            >>> import scipy.stats
            >>> import microstructpy as msp
            >>> S = scipy.stats.expon(scale=2)
            >>> S.mean()^2 + S.var()
            8.0
            >>> msp.geometry.Square.area_expectation(side_length=S)
            8.0

        Args:
            **kwargs: Keyword arguments, same as :class:`.Square` but the
                inputs can be from the :mod:`scipy.stats` module.

        Returns:
            float: Expected/average area of the square.

        """
        if 'side_length' in kwargs:
            len_dist = kwargs['side_length']

            try:
                area_exp = len_dist.moment(2)
            except AttributeError:
                area_exp = len_dist * len_dist
            return area_exp

        return Rectangle.area_expectation(**kwargs)

    # ----------------------------------------------------------------------- #
    # Circle Approximation                                                    #
    # ----------------------------------------------------------------------- #
    def approximate(self, x1=None):
        """Approximate square with a set of circles

        This method approximates a square with a set of circles.
        These circles are spaced uniformly along the edges of the square
        with distance ``x1`` between them.

        Example
        -------

        For a square with side_length=1, and x1=0.2,
        the approximation would look like :numref:`f_api_square_approx`.

        .. _f_api_square_approx:
        .. figure:: ../../auto_examples/geometry/images/sphx_glr_plot_rectangle_002.png

            Circular approximation of square.

        Args:
            x1 (float or None): *(optional)* Spacing between the circles.
                If not specified, the spacing is 0.25x the side length.

        Returns:
            numpy.ndarray: An Nx3 array, where each row is a circle and the
            columns are x, y, and r.

        """  # NOQA: E501
        return Rectangle.approximate(self, x1)
