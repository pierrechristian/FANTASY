# Copyright (C) 2020 Pierre Christian and Chi-kwan Chan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
################### USER GUIDE ###################
FANTASY is a geodesic integration code for arbitrary metrics with automatic differentiation. Please refer to Christian and Chan, 2020 for details.

################### Inputing the Metric ###################
Components of the metric are stored in the functions g00, g01, g02, etc that can be found under the heading "Metric Components". Each of these take as input a list called Param, which contains the fixed parameters of the metric (e.g., 'M' and 'a' for the Kerr metric in Boyer-Lindquist coordinates) and a list called Coord, which contains the coordinates (e.g., 'r' and 't' for the Kerr metric in Boyer-Lindquist coordinates). In order to set up a metric,
Step 1) Write down the fixed parameters in a list
Step 2) Write down the coordinates in a list
Step 3) Type the metric into the functions under "Metric Components".

Example: Kerr in Boyer-Lindquist coordinates
Step 1) The fixed parameters are listed as [M,a]
Step 2) The coordinates are listed as [t,r,theta,phi]
Step 3) Type in the metric components, for example, the g11 function becomes:

def g11(Param,Coord):
    return (Param[1]**2.-2.*Param[0]*Coord[1]+Coord[1]**2.)/(Coord[1]**2.+Param[1]**2.*cos(Coord[2])**2.)

Extra step) To make your code more readable, you can redefine variables in place of Param[i] or Coord[i], for example, the g11 function can be rewritten as:
def g11(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    return (a**2.-2.*M*r+r**2.)/(r**2.+a**2.*cos(theta)**2.)

################### A Guide on Choosing omega ###################
The parameter omega determines how much the two phase spaces interact with each other. The smaller omega is, the smaller the integration error, but if omega is too small, the equation of motion will become non-integrable. Thus, it is important to find an omega that is appropriate for the problem at hand. The easiest way to choose an omega is through trial and error:

Step 1) Start with omega=1; if you are working in geometric/code units in which all relevant factors are ~unity, this is usually already a good choice of omega
Step 2) If the trajectory varies wildly with time (this indicates highly chaotic, non-integrable behavior), increase omega and redo integration
Step 3) Repeat Step 2) until trajectory converges

################### Running the Code ###################
To run the code, run the function geodesic_integrator(N,delta,omega,q0,p0,Param,order). N is the number of steps, delta is the timestep, omega is the interaction parameter between the two phase spaces, q0 is a list containing the initial position, p0 is a list containing the initial momentum, Param is a list containing the fixed parameters of the metric (e.g., [M,a] for Kerr metric in Boyer-Lindquist coordinates), and order is the integration order. You can choose either order=2 for a 2nd order scheme or order=4 for a 4th order scheme.

################### Reading the Output ###################
The output is a numpy array indexed by timestep. For each timestep, the output contains four lists:

output[timestep][0] = a list containing the position of the particle at said timestep in the 1st phase space
output[timestep][1] = a list containing the momentum of the particle at said timestep in the 1st phase space
output[timestep][2] = a list containing the position of the particle at said timestep in the 2nd phase space
output[timestep][3] = a list containing the momentum of the particle at said timestep in the 2nd phase space

As long as the equation of motion is integrable (see section "A Guide on Choosing omega"), the trajectories in the two phase spaces will quickly converge, and you can choose either one as the result of your calculation.

'''

################### Code Preamble ###################

from pylab import *
from scipy import special
import numpy


class dual:
  def __init__(self, first, second):
    self.f = first
    self.s = second

  def __mul__(self,other):
    if isinstance(other,dual):
      return dual(self.f*other.f, self.s*other.f+self.f*other.s)
    else:
      return dual(self.f*other, self.s*other)

  def __rmul__(self,other):
    if isinstance(other,dual):
      return dual(self.f*other.f, self.s*other.f+self.f*other.s)
    else:
      return dual(self.f*other, self.s*other)

  def __add__(self,other):
    if isinstance(other,dual):
      return dual(self.f+other.f, self.s+other.s)
    else:
      return dual(self.f+other,self.s)

  def __radd__(self,other):
    if isinstance(other,dual):
      return dual(self.f+other.f, self.s+other.s)
    else:
      return dual(self.f+other,self.s)

  def __sub__(self,other):
    if isinstance(other,dual):
      return dual(self.f-other.f, self.s-other.s)
    else:
      return dual(self.f-other,self.s)

  def __rsub__(self, other):
    return dual(other, 0) - self

  def __truediv__(self,other):
    ''' when the first component of the divisor is not 0 '''
    if isinstance(other,dual):
      return dual(self.f/other.f, (self.s*other.f-self.f*other.s)/(other.f**2.))
    else:
      return dual(self.f/other, self.s/other)

  def __rtruediv__(self, other):
    return dual(other, 0).__truediv__(self)

  def __neg__(self):
      return dual(-self.f, -self.s)

  def __pow__(self, power):
    return dual(self.f**power,self.s * power * self.f**(power - 1))

  def sin(self):
    return dual(numpy.sin(self.f),self.s*numpy.cos(self.f))

  def cos(self):
    return dual(numpy.cos(self.f),-self.s*numpy.sin(self.f))

  def tan(self):
    return sin(self)/cos(self)

  def log(self):
    return dual(numpy.log(self.f),self.s/self.f)

  def exp(self):
    return dual(numpy.exp(self.f),self.s*numpy.exp(self.f))

def dif(func,x):
    funcdual = func(dual(x,1.))
    if isinstance(funcdual,dual):
        return func(dual(x,1.)).s
    else:
        ''' this is for when the function is a constant, e.g. gtt:=0 '''
        return 0

def dualtest(dual1,dual2):
    x = dual(dual1[0],dual1[1])
    y = dual(dual2[0],dual2[1])
    return 2.*x

def diftest(x):
    return dif(lambda y:cos(y)**2,x)

################### Metric Components ###################

# Diagonal components of the metric
def g00(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    Delta = r**2.-2.*M*r+a**2.
    rhosq = r**2.+a**2.*cos(theta)**2.
    return -(r**2.+a**2.+2.*M*r*a**2.*sin(theta)**2./rhosq)/Delta
def g11(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    return (a**2.-2.*M*r+r**2.)/(r**2.+a**2.*cos(theta)**2.)
def g22(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    return 1./(r**2.+a**2.*cos(theta)**2.)
def g33(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    Delta = r**2.-2.*M*r+a**2.
    rhosq = r**2.+a**2.*cos(theta)**2.
    return (1./(Delta*sin(theta)**2.))*(1.-2.*M*r/rhosq)

# Off-diagonal components of the metric
def g01(Param,Coord):
    return 0
def g02(Param,Coord):
    return 0
def g03(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    Delta = r**2.-2.*M*r+a**2.
    rhosq = r**2.+a**2.*cos(theta)**2.
    return -(2.*M*r*a)/(rhosq*Delta)
def g12(Param,Coord):
    return 0
def g13(Param,Coord):
    return 0
def g23(Param,Coord):
    return 0

################### Metric Derivatives ###################

def dm(Param,Coord,metric,wrt):
    ''' wrt = 0,1,2,3 '''
    point_d = Coord[wrt]

    point_0 = dual(Coord[0],0)
    point_1 = dual(Coord[1],0)
    point_2 = dual(Coord[2],0)
    point_3 = dual(Coord[3],0)

    if metric == 'g00':
        if wrt == 0:
            return dif(lambda p:g00(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g00(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g00(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g00(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g11':
        if wrt == 0:
            return dif(lambda p:g11(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g11(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g11(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g11(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g22':
        if wrt == 0:
            return dif(lambda p:g22(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g22(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g22(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g22(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g33':
        if wrt == 0:
            return dif(lambda p:g33(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g33(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g33(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g33(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g44':
        if wrt == 0:
            return dif(lambda p:g44(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g44(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g44(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g44(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g01':
        if wrt == 0:
            return dif(lambda p:g01(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g01(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g01(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g01(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g02':
        if wrt == 0:
            return dif(lambda p:g02(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g02(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g02(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g02(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g03':
        if wrt == 0:
            return dif(lambda p:g03(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g03(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g03(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g03(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g12':
        if wrt == 0:
            return dif(lambda p:g12(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g12(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g12(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g12(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g13':
        if wrt == 0:
            return dif(lambda p:g13(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g13(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g13(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g13(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g23':
        if wrt == 0:
            return dif(lambda p:g23(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g23(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g23(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g23(Param,[point_0,point_1,point_2,p]),point_d)

################### Integrator ###################

def Hamil_inside(q,p,Param,wrt):
    return p[0]*p[0]*dm(Param,q,'g00',wrt) +  p[1]*p[1]*dm(Param,q,'g11',wrt) +  p[2]*p[2]*dm(Param,q,'g22',wrt) +  p[3]*p[3]*dm(Param,q,'g33',wrt) +  2*p[0]*p[1]*dm(Param,q,'g01',wrt) +  2*p[0]*p[2]*dm(Param,q,'g02',wrt) + 2*p[0]*p[3]*dm(Param,q,'g03',wrt) +  2*p[1]*p[2]*dm(Param,q,'g12',wrt) +  2*p[1]*p[3]*dm(Param,q,'g13',wrt) + 2*p[2]*p[3]*dm(Param,q,'g23',wrt)

def phiHA(delta,omega,q1,p1,q2,p2,Param):
    ''' q1=(t1,r1,theta1,phi1), p1=(pt1,pr1,ptheta1,pphi1), q2=(t2,r2,theta2,phi2), p2=(pt2,pr2,ptheta2,pphi2) '''
    dq1H_p1_0 = 0.5*(Hamil_inside(q1,p2,Param,0))
    dq1H_p1_1 = 0.5*(Hamil_inside(q1,p2,Param,1))
    dq1H_p1_2 =  0.5*(Hamil_inside(q1,p2,Param,2))
    dq1H_p1_3 =  0.5*(Hamil_inside(q1,p2,Param,3))

    p1_update_array = numpy.array([dq1H_p1_0,dq1H_p1_1,dq1H_p1_2,dq1H_p1_3])
    p1_updated = p1 - delta*p1_update_array

    dp2H_q2_0 = g00(Param,q1)*p2[0] + g01(Param,q1)*p2[1] + g02(Param,q1)*p2[2] + g03(Param,q1)*p2[3]
    dp2H_q2_1 = g01(Param,q1)*p2[0] + g11(Param,q1)*p2[1] + g12(Param,q1)*p2[2] + g13(Param,q1)*p2[3]
    dp2H_q2_2 = g02(Param,q1)*p2[0] + g12(Param,q1)*p2[1] + g22(Param,q1)*p2[2] + g23(Param,q1)*p2[3]
    dp2H_q2_3 = g03(Param,q1)*p2[0] + g13(Param,q1)*p2[1] + g23(Param,q1)*p2[2] + g33(Param,q1)*p2[3]

    q2_update_array = numpy.array([dp2H_q2_0,dp2H_q2_1,dp2H_q2_2,dp2H_q2_3])
    q2_updated = q2 + delta*q2_update_array

    return (q2_updated, p1_updated)

def phiHB(delta,omega,q1,p1,q2,p2,Param):
    ''' q1=(t1,r1,theta1,phi1), p1=(pt1,pr1,ptheta1,pphi1), q2=(t2,r2,theta2,phi2), p2=(pt2,pr2,ptheta2,pphi2) '''
    dq2H_p2_0 = 0.5*(Hamil_inside(q2,p1,Param,0))
    dq2H_p2_1 = 0.5*(Hamil_inside(q2,p1,Param,1))
    dq2H_p2_2 =  0.5*(Hamil_inside(q2,p1,Param,2))
    dq2H_p2_3 =  0.5*(Hamil_inside(q2,p1,Param,3))

    p2_update_array = numpy.array([dq2H_p2_0,dq2H_p2_1,dq2H_p2_2,dq2H_p2_3])
    p2_updated = p2 - delta*p2_update_array

    dp1H_q1_0 = g00(Param,q2)*p1[0] + g01(Param,q2)*p1[1] + g02(Param,q2)*p1[2] + g03(Param,q2)*p1[3]
    dp1H_q1_1 = g01(Param,q2)*p1[0] + g11(Param,q2)*p1[1] + g12(Param,q2)*p1[2] + g13(Param,q2)*p1[3]
    dp1H_q1_2 = g02(Param,q2)*p1[0] + g12(Param,q2)*p1[1] + g22(Param,q2)*p1[2] + g23(Param,q2)*p1[3]
    dp1H_q1_3 = g03(Param,q2)*p1[0] + g13(Param,q2)*p1[1] + g23(Param,q2)*p1[2] + g33(Param,q2)*p1[3]

    q1_update_array = numpy.array([dp1H_q1_0,dp1H_q1_1,dp1H_q1_2,dp1H_q1_3])
    q1_updated = q1 + delta*q1_update_array

    return (q1_updated, p2_updated)

def phiHC(delta,omega,q1,p1,q2,p2,Param):
    q1 = numpy.array(q1)
    q2 = numpy.array(q2)
    p1 = numpy.array(p1)
    p2 = numpy.array(p2)

    q1_updated = 0.5*( q1+q2 + (q1-q2)*numpy.cos(2.*omega*delta) + (p1-p2)*numpy.sin(2.*omega*delta) )
    p1_updated = 0.5*( p1+p2 + (p1-p2)*numpy.cos(2.*omega*delta) - (q1-q2)*numpy.sin(2.*omega*delta) )

    q2_updated = 0.5*( q1+q2 - (q1-q2)*numpy.cos(2.*omega*delta) - (p1-p2)*numpy.sin(2.*omega*delta) )
    p2_updated = 0.5*( p1+p2 - (p1-p2)*numpy.cos(2.*omega*delta) + (q1-q2)*numpy.sin(2.*omega*delta) )

    return (q1_updated, p1_updated, q2_updated, p2_updated)

def updator(delta,omega,q1,p1,q2,p2,Param):
    first_HA_step = numpy.array([q1, phiHA(0.5*delta,omega,q1,p1,q2,p2,Param)[1], phiHA(0.5*delta,omega,q1,p1,q2,p2,Param)[0], p2])
    first_HB_step = numpy.array([phiHB(0.5*delta,omega,first_HA_step[0],first_HA_step[1],first_HA_step[2],first_HA_step[3],Param)[0], first_HA_step[1], first_HA_step[2], phiHB(0.5*delta,omega,first_HA_step[0],first_HA_step[1],first_HA_step[2],first_HA_step[3],Param)[1]])
    HC_step = phiHC(delta,omega,first_HB_step[0],first_HB_step[1],first_HB_step[2],first_HB_step[3],Param)
    second_HB_step = numpy.array([phiHB(0.5*delta,omega,HC_step[0],HC_step[1],HC_step[2],HC_step[3],Param)[0], HC_step[1], HC_step[2], phiHB(0.5*delta,omega,HC_step[0],HC_step[1],HC_step[2],HC_step[3],Param)[1]])
    second_HA_step = numpy.array([second_HB_step[0], phiHA(0.5*delta,omega,second_HB_step[0],second_HB_step[1],second_HB_step[2],second_HB_step[3],Param)[1], phiHA(0.5*delta,omega,second_HB_step[0],second_HB_step[1],second_HB_step[2],second_HB_step[3],Param)[0], second_HB_step[3]])

    return second_HA_step

def updator_4(delta,omega,q1,p1,q2,p2,Param):
    z14 = 1.3512071919596578
    z04 = -1.7024143839193155
    step1 = updator(delta*z14,omega,q1,p1,q2,p2,Param)
    step2 = updator(delta*z04,omega,step1[0],step1[1],step1[2],step1[3],Param)
    step3 = updator(delta*z14,omega,step2[0],step2[1],step2[2],step2[3],Param)

    return step3

def geodesic_integrator(N,delta,omega,q0,p0,Param,order=2):
    q1=q0
    q2=q0
    p1=p0
    p2=p0

    result_list = []
    result = (q1,p1,q2,p2)

    for count, timestep in enumerate(range(N)):
        if order == 2:
            print('Integration order = 2')
            updated_array = updator(delta,omega,result[0],result[1],result[2],result[3],Param)
        elif order == 4:
            print('Integration order = 4')
            updated_array = updator_4(delta,omega,result[0],result[1],result[2],result[3],Param)

        result = updated_array
        result_list += [result]

    return result_list
