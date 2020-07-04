"""Miscellaneous functions

This private module contains miscellaneous functions.
"""
import numpy as np

__author__ = 'Kenneth (Kip) Hart'

kw_solid = {'crystalline', 'granular', 'solid'}
kw_amorph = {'amorphous', 'glass', 'matrix'}
kw_void = {'void', 'crack', 'hole'}

ori_kws = {'orientation', 'matrix', 'angle', 'angle_deg', 'angle_rad',
           'rot_seq', 'rot_seq_rad', 'rot_seq_deg'}
gen_kws = {'material_type', 'fraction', 'shape', 'name', 'color', 'position'}

demo_needs = {'basalt_circle.xml': ['aphanitic_cdf.csv', 'olivine_cdf.csv'],
              'from_image.py': ['aluminum_micro.png']}

mpl_plural_kwargs = {'edgecolors', 'facecolors', 'linewidths', 'antialiaseds',
                     'offsets'}
plt_3d_adj = {
    'left': 0.4,
    'right': 1,
    'bottom': 0,
    'top': 0.8,
}


# --------------------------------------------------------------------------- #
#                                                                             #
# Convert String to Value (Infer Type)                                        #
#                                                                             #
# --------------------------------------------------------------------------- #
def from_str(string):
    """ Convert string to number

    This function takes a string and converts it into a number or a list.

    Args:
        string (str): The string.

    Returns:
        The value in the string.

    """
    beg_delims = ('(', '[', '{', '<')
    end_delims = (')', ']', '}', '>')

    string = string.strip()
    if any([c in string for c in beg_delims + end_delims]) or ',' in string:
        val = _list_from_str(string, beg_delims, end_delims)
    else:
        val = _single_from_str(string)
    return val


def _list_from_str(string, beg_delims, end_delims):
    if string[0] in beg_delims:
        string = string[1:]
    if string[-1] in end_delims:
        string = string[:-1]
    val = []
    n_beg = 0
    n_end = 0
    elem_str = ''
    for char in string:
        if char in beg_delims:
            n_beg += 1
        elif char in end_delims:
            n_end += 1

        if (char == ',') and n_beg == n_end:
            val.append(from_str(elem_str.strip()))
            elem_str = ''
        else:
            elem_str += char
    val.append(from_str(elem_str.strip()))
    return val


def _single_from_str(string):
    try:
        val = int(string)
    except ValueError:
        try:
            val = float(string)
        except ValueError:
            if string.lower() in ('true', 'yes'):
                val = True
            elif string.lower() in ('false', 'no'):
                val = False
            else:
                val = str(string)
    return val


def axisEqual3D(ax):
    '''From stackoverflow: https://stackoverflow.com/a/19248731'''
    extents = np.array([getattr(ax, 'get_{}lim'.format(d))() for d in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def ax_objects(ax):
    '''Number of objects in plot.'''
    n = 0
    for att in ['collections', 'images', 'lines', 'patches', 'texts']:
        n += len(getattr(ax, att))
    return n
