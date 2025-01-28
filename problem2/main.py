#!/usr/bin/env python

from numpy import exp, log, sin, cos, arccos, pi, linspace, meshgrid, full_like, array, cross, outer
from matplotlib import pyplot as plt

def cartesian_coord(theta, phi):
	return (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))

# a
def intersect_sphere(ax, f1, f2):
	t = linspace(-1, 1, 101)
	x1, y1, z1 = cartesian_coord(*f1(t))
	x2, y2, z2 = cartesian_coord(*f2(t))
	ax.plot(x1, y1, z1, color='b')
	ax.plot(x2, y2, z2, color='b')
	cx, cy, cz = x1[50], y1[50], z1[50]
	tan1 = array([x1[51]-cx, y1[51]-cy, z1[51]-cz])
	tan1 /= (tan1**2).sum()**0.5
	tan2 = array([x2[51]-cx, y2[51]-cy, z2[51]-cz])
	tan2 /= (tan2**2).sum()**0.5
	tx1, ty1, tz1 = tan1
	tx2, ty2, tz2 = tan2
	ax.plot([cx, cx+tx1/2], [cy, cy+ty1/2], [cz, cz+tz1/2], color='r')
	ax.plot([cx, cx+tx2/2], [cy, cy+ty2/2], [cz, cz+tz2/2], color='r')
	angle = arccos((tan1*tan2).sum())
	ax.text(cx, cy, cz, f'{angle:.2f}')

def intersect_stereo(ax, f1, f2):
	t = linspace(-1, 1, 101)
	x1, y1, z1 = cartesian_coord(*f1(t))
	x2, y2, z2 = cartesian_coord(*f2(t))
	x1, y1 = x1/(1-z1), y1/(1-z1)
	x2, y2 = x2/(1-z2), y2/(1-z2)
	ax.plot(x1, y1, color='b')
	ax.plot(x2, y2, color='b')
	cx, cy = x1[50], y1[50]
	tan1 = array([x1[51]-cx, y1[51]-cy])
	tan1 /= (tan1**2).sum()**0.5
	tan2 = array([x2[51]-cx, y2[51]-cy])
	tan2 /= (tan2**2).sum()**0.5
	tx1, ty1 = tan1
	tx2, ty2 = tan2
	ax.plot([cx, cx+tx1/2], [cy, cy+ty1/2], color='r')
	ax.plot([cx, cx+tx2/2], [cy, cy+ty2/2], color='r')
	angle = arccos((tan1*tan2).sum())
	ax.text(cx, cy, f'{angle:.2f}')

theta = linspace(0, pi, 20)
phi = linspace(0, 2*pi, 40)
theta, phi = meshgrid(theta, phi)
f1 = lambda t: (pi/2+pi/4*t, pi/16*t)
f2 = lambda t: (pi/2+pi/16*t, pi/4*t)
f3 = lambda t: (pi/4+pi/4*(exp(t)-1), pi+pi/4*t)
f4 = lambda t: (pi/4-pi/16*t, pi+pi/4*log(1+t/2))

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(*cartesian_coord(theta, phi), color='grey')
intersect_sphere(ax, f1, f2)
intersect_sphere(ax, f3, f4)
plt.show()

fig, ax = plt.subplots()
x, y, z = cartesian_coord(theta, phi)
intersect_stereo(ax, f1, f2)
intersect_stereo(ax, f3, f4)
ax.set_xlabel("$x'$")
ax.set_ylabel("$y'$")
plt.show()

# b
theta = linspace(0, pi, 20)
phi = linspace(0, 2*pi, 40)
theta, phi = meshgrid(theta, phi)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(*cartesian_coord(theta, phi), color='grey')

def great_circle(normal_vec):
	normal_vec /= (array(normal_vec, dtype='float64')**2).sum()**0.5
	p1 = array([-normal_vec[1], normal_vec[0], 0])
	p1 /= (p1**2).sum()**0.5
	p2 = cross(normal_vec, p1)
	psi = linspace(0, 2*pi, 100)
	return outer(p1, cos(psi)) + outer(p2, sin(psi))

ax.plot(*great_circle((1,2,3)), color='r')
ax.plot(*great_circle((1,1,-1)), color='g')
ax.plot(*great_circle((5,-2,8)), color='b')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.show()

def stereo_circle(normal_vec):
	x, y, z = great_circle(normal_vec)
	return x/(1-z), y/(1-z)

plt.plot(*stereo_circle((1,2,3)), color='r')
plt.plot(*stereo_circle((1,1,-1)), color='g')
plt.plot(*stereo_circle((5,-2,8)), color='b')
plt.xlabel("$x'$")
plt.ylabel("$y'$")
plt.show()

# c
def stereo(theta, phi):
	return sin(theta)*cos(phi)/(1-cos(theta)), sin(theta)*sin(phi)/(1-cos(theta))

def stereo_vector(theta, phi, x_theta, x_phi):
	x_x = (-cos(phi)*x_theta - sin(theta)*sin(phi)*x_phi) / (1-cos(theta))
	x_y = (-sin(phi)*x_theta + sin(theta)*cos(phi)*x_phi) / (1-cos(theta))
	return x_x, x_y

def transport(theta, phi, x0_theta, x0_phi):
	x_theta = x0_theta*cos(phi*cos(theta)) + x0_phi*sin(theta)*sin(phi*cos(theta))
	x_phi = x0_phi*cos(phi*cos(theta)) - x0_theta/sin(theta)*sin(phi*cos(theta))
	return x_theta, x_phi

phi = linspace(0, 2*pi, 20)
theta = full_like(phi, pi/4)
plt.plot(*stereo(theta, phi), color='b')
plt.quiver(*stereo(theta, phi), *stereo_vector(theta, phi, *transport(theta, phi, 0.2, 0.1)), color='r')
plt.quiver(*stereo(theta, phi), *stereo_vector(theta, phi, *transport(theta, phi, 0.1, 0.2)), color='g')
plt.xlabel("$x'$")
plt.ylabel("$y'$")
plt.show()

# d
phi = linspace(0, 2*pi, 200)
theta = full_like(phi, pi/4)
x1_x, x1_y = stereo_vector(theta, phi, *transport(theta, phi, 0.2, 0.1))
x2_x, x2_y = stereo_vector(theta, phi, *transport(theta, phi, 0.1, 0.2))
plt.plot(phi, (x1_x*x2_x + x1_y*x2_y) * 4/(1+x**2+y**2)**2)
plt.xlabel("$\\phi$")
plt.ylabel("Inner product")
plt.show()

# Yes, it preserves inner product, as the plot is a constant.
# This is obvious because coordinate transformation preserves inner product.

# f
# No, because the holonomy is a coordinate-independent concept.
