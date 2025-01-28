#!/usr/bin/env python

from numpy import sin, cos, pi, linspace, meshgrid, full_like, gradient, stack, array, zeros_like
from matplotlib import pyplot as plt

# a
def spherical_coord(theta, phi):
	return (full_like(theta, 1), theta, phi)

def cartesian_coord(theta, phi):
	return (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))

def cylindrical_coord(theta, phi):
	return (sin(theta), phi, cos(theta))

# b
def e_r(theta, phi):
	return cartesian_coord(theta, phi)
def e_theta(theta, phi):
	return cartesian_coord(theta+pi/2, phi)
def e_phi(theta, phi):
	return (-sin(theta)*sin(phi), sin(theta)*cos(phi), zeros_like(theta))

theta = linspace(0, pi/2, 5)
phi = linspace(0, 2*pi, 20)
theta, phi = meshgrid(theta, phi)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_surface(*cartesian_coord(theta, phi))
ax.quiver(*cartesian_coord(theta, phi), *e_r(theta, phi), length=0.1, color='r')
ax.quiver(*cartesian_coord(theta, phi), *e_theta(theta, phi), length=0.1, color='g')
ax.quiver(*cartesian_coord(theta, phi), *e_phi(theta, phi), length=0.1, color='b')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.show()

# c
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_surface(*spherical_coord(theta, phi))
ax.set_xlabel('$r$')
ax.set_ylabel(r'$\theta$')
ax.set_zlabel(r'$\phi$')
plt.show()

# d
def local_coord(f, x, y):
	dx = x[1:,:] - x[:-1,:]
	dy = y[:,1:] - y[:,:-1]
	z = f(x, y)
	dfdx = gradient(z, dx, axis=0)
	dfdy = gradient(z, dy, axis=1)
	norms = (dfdx**2 + dfdy**2 + 1)**0.5
	return stack((dfdx/norms, dfdy/norms, -1/norms), axis=-1)

# e
theta = linspace(0, pi, 20)
phi = linspace(0, 2*pi, 40)
theta, phi = meshgrid(theta, phi)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(*cartesian_coord(theta, phi))

theta = linspace(pi/5, pi/2, 10)
phi = full_like(theta, 0)
x_theta = full_like(theta, 0.1)
x_phi = full_like(theta, 0.1)
ax.quiver(*cartesian_coord(theta, phi), *(x_theta*array(e_theta(theta,phi))+x_phi*array(e_phi(theta,phi))), color='r')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.show()

# f
theta = linspace(0, pi, 20)
phi = linspace(0, 2*pi, 40)
theta, phi = meshgrid(theta, phi)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(*cartesian_coord(theta, phi))

phi = linspace(0, 2*pi, 20)
theta = full_like(phi, pi/4)
x0_theta = 0.1
x0_phi = 0.1
x_theta = x0_theta*cos(phi*cos(theta)) + x0_phi*sin(theta)*sin(phi*cos(theta))
x_phi = x0_phi*cos(phi*cos(theta)) - x0_theta/sin(theta)*sin(phi*cos(theta))
ax.quiver(*cartesian_coord(theta, phi), *(x_theta*array(e_theta(theta,phi))+x_phi*array(e_phi(theta,phi))), color='r')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.show()

# g
theta = linspace(0, pi, 200)
strength = cos(2*pi*cos(theta))
plt.plot(theta, strength)
plt.xlabel(r'$\theta_0$')
plt.ylabel('strength')
plt.show()
