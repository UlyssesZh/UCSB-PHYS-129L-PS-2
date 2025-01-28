#!/usr/bin/env python

from numpy import pi, linspace, array, unique, argmin, argmax, cross, loadtxt, vstack, pad, isclose
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

# a
def quickhull(points):
	def add_hull(points, p1, p2):
		if not len(points):
			return []
		distances = cross(points - p1, p2 - p1)
		farthest_point_index = argmax(distances)
		farthest_point = points[farthest_point_index]
		left_set_1 = points[cross(points - p1, farthest_point - p1) > 0]
		left_set_2 = points[cross(points - farthest_point, p2 - farthest_point) > 0]
		return add_hull(left_set_1, p1, farthest_point) + [farthest_point] + add_hull(left_set_2, farthest_point, p2)
	points = unique(points, axis=0)
	if len(points) < 3:
		return points
	leftmost = points[argmin(points[:, 0])]
	rightmost = points[argmax(points[:, 0])]
	above_set = points[cross(points - leftmost, rightmost - leftmost) > 0]
	below_set = points[cross(points - leftmost, rightmost - leftmost) < 0]
	upper_hull = add_hull(above_set, leftmost, rightmost)
	lower_hull = add_hull(below_set, rightmost, leftmost)
	return array([leftmost] + upper_hull + [rightmost] + lower_hull)

def triangle_area(p1, p2, p3):
	p1 = pad(p1, (0, 3-len(p1)), 'constant')
	p2 = pad(p2, (0, 3-len(p2)), 'constant')
	p3 = pad(p3, (0, 3-len(p3)), 'constant')
	return ((cross(p2 - p1, p3 - p1)/2)**2).sum()**0.5

points = loadtxt('mesh.dat', skiprows=1)
hull = quickhull(points)
hull = vstack((hull, hull[0]))
plt.plot(hull[:, 0], hull[:, 1], color='r')

triangles = Delaunay(points).simplices
# triangles = [v for v in triangles if not isclose(triangle_area(*points[v]), 0)]
plt.triplot(points[:, 0], points[:, 1], triangles, color='g')

plt.scatter(points[:, 0], points[:, 1], color='b')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

# b
def surface(x, y):
	return x**2+y**2

colors = []
for triangle in triangles:
	area = triangle_area(*points[triangle])
	lifted_area = triangle_area(*[[*p, surface(*p)] for p in points[triangle]])
	colors.append(area/lifted_area)
tpc = plt.tripcolor(points[:, 0], points[:, 1], facecolors=colors, triangles=triangles, cmap='viridis')
plt.colorbar(tpc)
plt.scatter(points[:, 0], points[:, 1], color='b')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

# c
# g_xx = 1 + 4 x^2
# g_yy = 1 + 4 y^2
# g_xy = g_yx = 4 x y

# d
