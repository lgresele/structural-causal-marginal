import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def make_rectangle(min_A, max_A, min_B, max_B):
    rectangle = np.array([[min_A, min_B], [min_A, max_B], [max_A, min_B], [max_A, max_B]])
    return rectangle

    plt.close()

def generate_polygon(points, label=None, color='red', edgecolor='black', k=1.0, alpha=0.2, capstyle='round', hatch=''):
    '''
    Input:
    points: array NxM, 
    
    Output:
    poly: matplotlib.patches.Polygon
    '''
    hull = ConvexHull(points)
    cent = np.mean(points, 0)
    pts = []
    for pt in points[hull.simplices]:
        pts.append(pt[0].tolist())
        pts.append(pt[1].tolist())

    pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                    p[0] - cent[0]))
    pts = pts[0::2]  # Deleting duplicates
    pts.insert(len(pts), pts[0])
    poly = Polygon(k*(np.array(pts)- cent) + cent,
                   facecolor=color, alpha=alpha, label=label, edgecolor=edgecolor, hatch=hatch)
    poly.set_capstyle(capstyle)
    return poly