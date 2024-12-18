# Fast Heat Equation
Project for HPPL course in 2024

## Authors

Volkov Daniil, Ildar Saiapov

## Overview

This repository contains our project for the High Performance Python Lab course. The focus is on implementing and accelerating the solution to the 3D heat equation using advanced computational techniques. By leveraging finite difference schemes and evolutionary factorization, we aim to demonstrate significant performance improvements in solving this problem.

## Heat equation

We solve:

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right)
$$

with initail and boundary conditions

## Evolutionary factorization

$$\left( E-\frac{\tau}{2}\Lambda \right) \frac{\hat{u}-u}{\tau} = \Lambda u + f\left(t+\frac{\tau}{2}, x,y,z \right)$$

$$\left( E-\frac{\tau}{2}\Lambda_z \right) \omega = (\Lambda_x+\Lambda_y+\Lambda_z) u + f\left(t+\frac{\tau}{2}, x,y,z \right)$$

$$\left( E-\frac{\tau}{2}\Lambda_y \right) v = \omega$$

$$\left( E-\frac{\tau}{2}\Lambda_x \right) \frac{\hat{u}-u}{\tau} = v$$

This scheme allow us to speed up computation.

## Comparison

To compare the speed of our algorithm, we implemented schemes with finite differences (with and without numba acceleration).

## Results

- Numba (finite differences) gave us acceleration in 23 times (compare finite differences).
- Factorization on cpu gave us acceleration in 97 times (compare finite differences).
- Factorization on gpu gave us acceleration in 271 times (compare finite differences).
