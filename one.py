from __future__ import division
# Georgia State University
# Neuroscience Institue
# Ricardo Erazo
# traveling waves in HH endogenous oscillatory neurons.
# dynamic variable decaying synapse implementation. Parallel simulation

#the purpose of this script is to provide a brief inhibitory pulse to the neurons at hand
#and measure its response. This is used to figure out the most appropriate way to shock
#neurons to allow complete depolarization burst.
#A second purpose of this script is to note the difference in burst duration as a function of
#parameter K2theta

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

nr_neurons = 3
delta1 = 1e-2
sigma1 = 2e-2
gsyn= 1
domain = np.arange(0,nr_neurons)*delta1
domain = np.array([domain])
Vt = -0.01
tau2 = 0.1
tspan = [0,13]
#tspan = [0,50]
delta2 = 0.1

neuronSpace = np.arange(0,nr_neurons*delta1,delta1)

J = np.zeros((nr_neurons,nr_neurons))
for i in np.arange(0,nr_neurons,1):
	J[:,i] = abs(neuronSpace - neuronSpace[i])
J = delta1*np.exp(-J/sigma1)/2/sigma1
J = J-np.diag(np.diag(J))
for i in np.arange(0,nr_neurons,1):
	J[i,i:nr_neurons] = 0
W = gsyn*J
W1 = W[0,:]
#Experimental parameters
Htheta= 0.04134
#Htheta= 0.041325
#K2theta= -0.0060
#K2theta= -0.0065
#K2theta= -0.0070
#K2theta= -0.0075
#K2theta= -0.008
#K2theta= -0.0085
K2theta= -0.0088
#K2theta= -0.009 #duty cycle too active. 0.5 approx

#Cell parameters
gNa= 105
ENa= 0.045
gK2= 30
EK=-0.07
gH= 4
EH= -0.021
gl= 8
El= -0.046
#Vrev= 0.01
#Vrev=-0.02
Vrev= -0.062

#time scales
Cm= 0.5
tau_h= 0.04050
tau_m= 0.1
tau_k= 2
tau_s= 0.1
tau1= Cm;
int_time = 50

# ODE
#initial conditions
## rest
v0= -0.0421996958
h0= 0.99223221
m0= 0.297443439
n0= 0.0152445526
## shocked
v2 = -4.4203449e-2
h2 = 9.92877246e-1
m2 = 3.02937855e-1
n2 = 1.52351209e-2

#initial conditions
y0 = np.array((v0,h0,m0,n0,0))
y2 = np.array((v2,h2,m2,n2,0))

initconds = np.zeros((1,nr_neurons*5))
initconds[0,0:nr_neurons] = v0
initconds[0,nr_neurons:nr_neurons*2] = h0
initconds[0,nr_neurons*2:nr_neurons*3] = m0
initconds[0,nr_neurons*3:nr_neurons*4] = n0
initconds[0,nr_neurons*4:nr_neurons*5] = 0
initconds[0,0] = v2
initconds[0,nr_neurons] = h2
initconds[0,nr_neurons*2]= m2
initconds[0,nr_neurons*3] = n2
initconds[0,nr_neurons*4] = 0
#y0 = np.reshape(initconds, (5,nr_neurons))
#for i in range(0,nr_neurons):
#    print (y0)
def evolve(t, y0):
    u = np.reshape(y0, (5,nr_neurons))
    
#    v = u[0,:]
#    h = u[1,:]
#    m = u[2,:]
#    n = u[3,:]
    s = u[4,:]
##    print(u,v)
    dy = np.zeros((nr_neurons,5))
#    print(dy[1,:])
    for i in range(0,nr_neurons):
#        print(y0)
        W0 = W[i,:]
        neuron = u[0:4,i]
        v,h,m,n= neuron
#        print (neuron)
#        print(i, v,h,m,n,s)
#        dy = np.zeros((nr_neurons,5))
        mNass=1./(1.+np.exp(-150.*(v+0.0305)))
        mK2ss=1./(1.+np.exp(-83.*(v+K2theta)))
        dv = (-1/Cm) * (gNa*mNass*mNass*mNass*h*(v-ENa)+ gK2*n*n*(v-EK) + gH*m*m*(v-EH) + gl*(v-El) + 0.006 +np.dot(W0,s)*(v-Vrev))
        dh = (1/(1+ np.exp(500*(v+0.0325))) - h)/tau_h
        dm = (1/(1+2*np.exp(180*(v+Htheta))+np.exp(500*(v+Htheta))) -m)/tau_m
        dn = (mK2ss - n)/tau_k
#        if v>Vt:
##        aa = np.transpose(np.nonzero(v>Vt))
#            s[i] = s[i]+delta2
##            spikes
#        if s[i]>1:
#            s[i] =1
#        
        ds = -s[i]/tau2
        ff = dv,dh,dm,dn,ds
#        print(ff,dy[i,0:5])
        dy[i,0:5] = ff
#        print (dv)
#        dy[i,0]= dv
#        dy[i,1]= dh
#        dy[i,2]= dm
#        dy[i,3]= dn
#        dy[i,4]= ds
#        print(i,dy)
#        dy[i,:] = dv,dh,dm,dn,ds
#    print(dy)
#    print(np.shape(dy))
    dv_vec = dy[:,0]
    dh_vec = dy[:,1]
    dm_vec = dy[:,2]
    dn_vec = dy[:,3]
    ds_vec = dy[:,4]
#    f = dv_vec, dh_vec, dm_vec, dn_vec, ds_vec
##    f = np.reshape(dy, (nr_neurons,5))
#    f1 = 1
#    print(f)
    temp1 = np.concatenate((dv_vec,dh_vec),axis=0)
    temp2 = np.concatenate((dm_vec,dn_vec,ds_vec),axis=0)
    f = np.concatenate((temp1,temp2),axis=0)
    return f
#    temp1 = np.concatenate((dv,dh),axis=0)
#    temp2 = np.concatenate((dm,dn,ds),axis=0)
#    f = np.concatenate((temp1,temp2),axis=0)
#    return f

def Vt_cross_ctr(t,y0): return y0[ctr]-Vt
#Vt_cross_ctr.terminal= True
Vt_cross_ctr.terminal= False
Vt_cross_ctr.direction = 1

#def Vt_cross_any(t,y0): return y0[0]-Vt
#    voltages = y0[0:nr_neurons]
#    print(voltages)
#    return voltages-Vt
#Vt_cross_any.terminal= True
Vt_cross_ctr.terminal= False
#Vt_cross_any.direction = 1

hmin = 1e-4
tev = np.arange(tspan[0],tspan[1],hmin)
spike_times = np.zeros((1,nr_neurons))
#for ctr in range(0,nr_neurons):
	#plt.figure(figsize=(10,10))
ctr = 0

#for nn in range(0,nr_neurons):
neural_net = solve_ivp(evolve, tspan,initconds[0],method='RK45',t_eval = tev,events =Vt_cross_ctr ,atol=1e-7,rtol=1e-5)

	#time_spike= neural_net.t_events[0]
	#spike_times[0,ctr]=neural_net.t_events[0]
	#print('Wave'+str(ctr),spike_times[0,ctr])

fgsz = 9
plt.figure(figsize=(fgsz,fgsz))
for i in range(0,nr_neurons):
		plt.plot(neural_net.t,neural_net.y[i,:]-0.1*i)
plt.savefig('V_g'+str(gsyn)+'.png')
plt.figure(figsize=(fgsz,fgsz))
for i in range(0,nr_neurons):
		plt.plot(neural_net.t,neural_net.y[i+nr_neurons*4,:]-0.1*i)
plt.title('K2theta='+str(K2theta)+'_gsyn'+str(gsyn))
plt.savefig('S_g'+str(gsyn)+'.png')
np.save('s_g'+str(gsyn),neural_net.y[nr_neurons*4,:])
np.save('v_g'+str(gsyn),neural_net.y[0,:])
np.save('t_g'+str(gsyn),neural_net.t)
