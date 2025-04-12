# ME700-ASSIGNMENT4

# Set Up

# Poiseuille Flow


Fuid flow through a channel is known as Poiseuille flow, shown in the figure. In this case, the horizontal flow is driven by a pressure difference across the channel (left to right).

__CHANNEL FLOW IMAGE__



## Navier Stokes Equation

The incompressible Naview Stoke equations is given as

$$\rho \left(\frac{\partial u}{\partial t}+u\cdot \nabla u \right) = \nabla \cdot \sigma (u, p) + f$$  
$$ \nabla\cdot u = 0$$  

The first equation consdiers conservation of momentum, with the terms on the right hand side being the stress tensor and the body forces (e.g. pressure). The second equation is tthe continuity equation. When considering Poiseuille flow, these can be scaled based on the problem geometry. Detail on scaling can be found in [1]

$$\rho \left(\frac{\partial u}{\partial t}+\text{Re} u\cdot u \right) = -\nabla p + \nabla ^2 u$$  
$$ \nabla\cdot u = 0$$  
$$ Re = \frac{\rho UH}{\mu}$$

## Assumptions

Assume the fluid is incompressible . In this problem, we will consider a 2D channel, so the velocity has $x$ and $y$ components.  
$$u = (u_x,u_y)$$  
For a horizontal channel, there is only horizontal flow. The velocity ($u_x$) is not dependent on $x$ because of the continuity equation.

$$ u = (u_x(x,y),0)$$
