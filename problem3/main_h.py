#!/usr/bin/env python

from numpy import array, unique, argmin, argmax, cross, loadtxt, stack, vstack, pad, full_like, dot
from numpy.linalg import inv, eig
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
points = points*0.4 - 2 # The points are in [0, 10], but we want them in [-2, 2]
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
	return x**2+x*y+y**2

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
# g_xx = 1 + (2 x + y)^2
# g_yy = 1 + (x + 2 y)^2
# g_xy = g_yx = (2 x + y) (x + 2 y)

# d
def triangle_normal(p1, p2, p3):
	normal = cross(p3 - p1, p2 - p1)
	return normal / (normal**2).sum()**0.5

def center(p1, p2, p3):
	return (p1 + p2 + p3) / 3

points3d = vstack((points.T, surface(*points.T))).T
centers = array([center(*points3d[v]) for v in triangles])
normals = array([triangle_normal(*points3d[v]) for v in triangles])

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(*points3d.T, triangles=triangles, cmap='viridis')
ax.quiver(*centers.T, *normals.T, color='r', length=0.5)
plt.show()

# e
vertex_normals = []
for i in range(len(points)):
	triangles_with_vertex = [t for t in triangles if i in t]
	normal = sum(triangle_normal(*points3d[t]) * triangle_area(*points3d[t]) for t in triangles_with_vertex)
	vertex_normals.append(normal / (normal**2).sum()**0.5)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(*points3d.T, triangles=triangles, cmap='viridis')
ax.quiver(*points3d.T, *array(vertex_normals).T, color='r', length=0.5)
plt.show()

# f
x = points3d[:, 0]
y = points3d[:, 1]
partial_x = array([full_like(x, 1), full_like(x, 0), 2*x+y]).T
partial_y = array([full_like(y, 0), full_like(y, 1), 2*y+x]).T
partial_xx = array([full_like(x, 0), full_like(x, 0), full_like(x, 2)]).T
partial_yy = array([full_like(y, 0), full_like(y, 0), full_like(y, 2)]).T
partial_xy = array([full_like(x, 0), full_like(x, 0), full_like(x, 1)]).T
ii_xx = (partial_xx*vertex_normals).sum(axis=1)
ii_yy = (partial_yy*vertex_normals).sum(axis=1)
ii_xy = (partial_xy*vertex_normals).sum(axis=1)
ii_yx = ii_xy
ii = stack([ii_xx, ii_xy, ii_yx, ii_yy], axis=-1).reshape(-1, 2, 2)

# g
p_xx = (partial_x*partial_x).sum(axis=1)
p_yy = (partial_y*partial_y).sum(axis=1)
p_xy = (partial_x*partial_y).sum(axis=1)
p_yx = p_xy
p = inv(stack([p_xx, p_xy, p_yx, p_yy], axis=-1).reshape(-1, 2, 2))
shape = array([dot(p[i], ii[i]) for i in range(len(points))])

principle = eig(shape).eigenvalues
gaussian = principle.prod(axis=-1)
mean = principle.sum(axis=-1)

plt.scatter(x, y, c=gaussian, cmap='viridis')
plt.colorbar(label='Gaussian curvature')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

plt.scatter(x, y, c=mean, cmap='viridis')
plt.colorbar(label='Mean curvature')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
