import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

class Simulation:
        def __init__(self,npoints=200,init_height=0.1,atwood=0.8,gravity=1.,alpha=0.,delta_square=0.01):
                self.vs=VortexSheet(npoints=npoints,init_height=init_height)
                self.vs.update(delta_square)
                self.A=atwood
                self.alpha=alpha
                self.g=gravity
                self.delta_square=delta_square
                
        def push_interface(self,dt):
                tx,ty=self.vs.unit_tangent()
                slide=0.5*self.alpha*self.vs.gamma
                self.vs.x+=dt*(self.vs.u+slide*tx)
                self.vs.y+=dt*(self.vs.v+slide*ty)

        def push_vorticity(self,dt):
                tx,ty=self.vs.unit_tangent()
                ax=(self.vs.u-self.uold)/dt
                ay=(self.vs.v-self.vold)/dt
                ut=centered_diff(self.vs.arc,self.vs.u,self.vs.length)
                vt=centered_diff(self.vs.arc,self.vs.v,self.vs.length)
                
                t1=2.*self.A*(tx*ax+ty*ay)
                t2=centered_diff(self.vs.arc,0.25*(self.alpha+self.A)*self.vs.gamma**2+2.*self.A*self.g*self.vs.y,self.vs.length)
                t3=-(1+self.alpha*self.A)*self.vs.gamma*(ut*tx+vt*ty)
                self.vs.gamma+=dt*(t1+t2+t3)

        def evolve(self,dt):
                self.save_velocity()    # save velocity before next step
                self.push_interface(dt)  # push points on interface
                self.vs.update(self.delta_square)       # update arc, length, velocity with new interface
                self.push_vorticity(dt)  # push vorticity
                self.vs.update_velocity(self.delta_square)  # update velocity

        def save_velocity(self):
                self.uold=self.vs.u.copy()
                self.vold=self.vs.v.copy()        
                

class VortexSheet:
        """ we assume periodic boundary over x\in[0,2\pi). """
        
        def __init__(self,npoints,init_height=1.):
                """ on periodic boundary # of points is equal to # of nintervals. """
                self.n=npoints
                
                self.x=np.linspace(0,2*np.pi,self.n,endpoint=False)
                self.y=init_height*np.cos(self.x)

                self.gamma=np.zeros(self.n)
                
                self.arc=np.zeros(self.n)
                self.length=1.
                self.u=np.zeros(self.n)
                self.v=np.zeros(self.n)                


        def update(self,delta_square):
                self.update_arclength()
                self.update_length()
                self.update_velocity(delta_square)
                
        def update_arclength(self):
                """ this is a vector of length n with zero in the first entry """
                dx=np.diff(self.x)
                dy=np.diff(self.y)
                self.arc[0]=0
                self.arc[1:]=np.cumsum(np.sqrt(dx**2+dy**2))
                
        def update_length(self):
                """ this is the total length of the periodic curve (add the segment back to the first point) """
                dx=np.diff(self.x)
                dy=np.diff(self.y)
                self.length=np.sum(np.sqrt(dx**2+dy**2))
                self.length+=np.sqrt((self.x[0]+2*np.pi-self.x[-1])**2+(self.y[0]-self.y[-1])**2)

        def trapeze_weights(self):
                """ returns a vector of length n with segment weights accroding to trapezoidal rule """
                Ds=np.zeros(self.n)
                Ds[0]=0.5*(self.arc[1]-self.arc[0]+self.length-self.arc[-1])
                Ds[-1]=0.5*(self.length-self.arc[-2])
                Ds[1:-1]=0.5*(self.arc[2:]-self.arc[:-2])
                return Ds

        def update_velocity(self,delta_square):
                """ performs the convolution over regularised kernel and vortex density applying trapezoidal rule. Returns two vectors of length n for each component of the velocity field (u,v). """
                fac=0.25/np.pi
                Ds=self.trapeze_weights()
                for i in range(self.n):
                        xd=self.x[i]-self.x
                        yd=self.y[i]-self.y
                        
                        Gf=self.gamma*Ds/(np.cosh(yd)-np.cos(xd)+delta_square)
                        self.u[i]=-fac*np.sum(np.sinh(yd)*Gf)
                        self.v[i]= fac*np.sum(np.sin(xd)*Gf)
                        
        def unit_tangent(self):
                dx=centered_diff(self.arc,self.x,self.length)
                dy=centered_diff(self.arc,self.y,self.length)
                norm=np.sqrt(dx**2+dy**2)
                return dx/norm,dy/norm
        
def centered_diff(s,A,period):
        """ assume s[0]=0 and s[n]=period """
        D=np.zeros(A.size)
        D[0]=(A[1]-A[-1])/(s[1]+period-s[-1])
        D[-1]=(A[0]-A[-2])/(period-s[-2])
        D[1:-1]=(A[2:]-A[:-2])/(s[2:]-s[:-2])
        return D
        
                

