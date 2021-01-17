# FANTASY
FANTASY is a geodesic integration code for arbitrary metrics with automatic differentiation. Please refer to Christian and Chan, 2021 for details.

### Inputing the Metric 

Components of the metric are stored in the functions g00, g01, g02, etc that can be found under the heading "Metric Components". Each of these take as input a list called Param, which contains the fixed parameters of the metric (e.g., 'M' and 'a' for the Kerr metric in Boyer-Lindquist coordinates) and a list called Coord, which contains the coordinates (e.g., 'r' and 't' for the Kerr metric in Boyer-Lindquist coordinates). In order to set up a metric,  

**Step 1)** Write down the fixed parameters in a list  
**Step 2)** Write down the coordinates in a list  
**Step 3)** Type the metric into the functions under "Metric Components" 

*Example*: Kerr in Boyer-Lindquist coordinates  
**Step 1)** The fixed parameters are listed as [M,a]   
**Step 2)** The coordinates are listed as [t,r,theta,phi]  
**Step 3)** Type in the metric components, for example, the g11 function becomes:

```
 def g11(Param,Coord):
    return (Param[1]**2.-2.*Param[0]*Coord[1]+Coord[1]**2.)/(Coord[1]**2.+Param[1]**2.*cos(Coord[2])**2.)
```

**Extra step)** To make your code more readable, you can redefine variables in place of Param[i] or Coord[i], for example, the g11 function can be rewritten as:
```
def g11(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    
    return (a**2.-2.*M*r+r**2.)/(r**2.+a**2.*cos(theta)**2.)
 ```
    
### A Guide on Choosing omega 

The parameter omega determines how much the two phase spaces interact with each other. The smaller omega is, the smaller the integration error, but if omega is too small, the equation of motion will become non-integrable. Thus, it is important to find an omega that is appropriate for the problem at hand. The easiest way to choose an omega is through trial and error:    

**Step 1)** Start with omega=1; if you are working in geometric/code units in which all relevant factors are ~unity, this is usually already a good choice of omega  
**Step 2)** If the trajectory varies wildly with time (this indicates highly chaotic, non-integrable behavior), increase omega and redo integration  
**Step 3)** Repeat Step 2) until trajectory converges  

### Running the Code 

To run the code, run the function geodesic_integrator(N,delta,omega,q0,p0,Param,order). N is the number of steps, delta is the timestep, omega is the interaction parameter between the two phase spaces, q0 is a list containing the initial position, p0 is a list containing the initial momentum, Param is a list containing the fixed parameters of the metric (e.g., [M,a] for Kerr metric in Boyer-Lindquist coordinates), and order is the integration order. You can choose either order=2 for a 2nd order scheme or order=4 for a 4th order scheme.

### Reading the Output 

The output is a numpy array indexed by timestep. For each timestep, the output contains four lists:
```
output[timestep][0] = a list containing the position of the particle at said timestep in the 1st phase space
output[timestep][1] = a list containing the momentum of the particle at said timestep in the 1st phase space
output[timestep][2] = a list containing the position of the particle at said timestep in the 2nd phase space
output[timestep][3] = a list containing the momentum of the particle at said timestep in the 2nd phase space
```
As long as the equation of motion is integrable (see section "A Guide on Choosing omega"), the trajectories in the two phase spaces will quickly converge, and you can choose either one as the result of your calculation.

### Automatic Jacobian 
Input coordinate transformations for the 0th, 1st, 2nd, 3rd coordinate in functions CoordTrans0, CoordTrans1, CoordTrans2, CoordTrans3. As an example, coordinate transformation from Spherical Schwarzschild to Cartesian Schwarzschild has been provided.
