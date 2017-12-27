import matplotlib.pyplot as plt
import numpy as np

import weave
from weave import converters
from numpy.linalg import inv

from SpikingModel import *
from GIF import *

from Filter_Rect_LogSpaced import *

import Tools
from Tools import reprint
import scipy
from scipy import linalg, matrix

def null(A, eps=1e-15):
    """
    :param A: matrix object
    :param eps: tolerance
    :return:
    """
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

class GIF_Ca_Kinetic(GIF):
    """
     Generalized Integrate and Fire model with voltage-dependent calcium current (VDCC).
     Spike are produced stochastically with firing intensity:
     lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),
     where the membrane potential dynamics is given by:

     C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j) - g_Ca*p_O*(V-E_Ca),

     where  E_Ca : reversal potential associated with the VDCC
            g_Ca : maximal conductance
            m    : activation gating variable
            h    : inactivation gating variable

     The firing threshold V_T is given by:

     V_T = Vt_star + sum_j gamma(t-\hat t_j),

     and \hat t_j denote the spike times, as in the standard GIF model.
     """

    def __init__(self, dt=0.1):
        GIF.__init__(self, dt=dt)

        self.E_Ca = 0.0  # mV, reversal potential associated with the voltage-dependent calcium current
        self.g_Ca = 0.01  # uS, maximal conductance of VDCC

    def kV(V):
        return 0.2*np.exp(V/25.6)

    def k_V(V):
        return 1.5.e-3*np.exp(-V/18.4)

    def k_O(V):
        return 100.e-3*np.exp(-V/34.6)

    kO = 500.e-3
    kI = 7.e-3
    k_I = 0.21e-3

    ########################################################################################################
    # FUNCTIONS FOR SIMULATIONS
    ########################################################################################################

    def simulate_openprob(self, V):
        """
        :param V: vector of voltages for a given training
        :return: array of openprob
        """

        # Find initial state probabilities


        p_O = np.array(np.zeros(V.size), dtype="double")
        p_dt = self.dt
        p_N = V.size
        V = np.array(V, dtype="double")

        code = """
            #include <math.h>
            float dt = float(p_dt);
            int N    = int(p_N);
            float kO = 500.e-3;
            float kI = 7.e-3;
            float k_I = 0.21e-3;
            float f = 0.2;
            float h = 0.45;
            float kV, k_V, k_O;     
            float p_C0, p_I0, p_C1, p_I1, p_C2, p_I2, p_C3, p_I3, p_C4, p_I4, p_IO;       
            float dp_C0dt, dp_I0dt, dp_C1dt, dp_I1dt, dp_C2dt, dp_I2dt, dp_C3dt, dp_I3dt, dp_C4dt, dp_I4dt, dp_IOdt;       
            float dpred_C0dt, dpred_I0dt, dpred_C1dt, dpred_I1dt, dpred_C2dt, dpred_I2dt, dpred_C3dt, dpred_I3dt, dpred_C4dt, dpred_I4dt, dpred_IOdt;       
            float pred_C0, pred_I0, pred_C1, pred_I1, pred_C2, pred_I2, pred_C3, pred_I3, pred_C4, pred_I4, pred_IO, pred_O;       
            float rand_max  = float(RAND_MAX);
            
            //Random initialization of state probabilities
            p_C0 = 0.1*rand()/rand_max;
            p_I0 = 0.1*rand()/rand_max;
            p_C1 = 0.1*rand()/rand_max;
            p_I1 = 0.1*rand()/rand_max;
            p_C2 = 0.1*rand()/rand_max;
            p_I2 = 0.1*rand()/rand_max;
            p_C3 = 0.1*rand()/rand_max;
            p_I3 = 0.1*rand()/rand_max;
            p_C4 = 0.1*rand()/rand_max;
            p_I4 = 0.1*rand()/rand_max;
            p_IO = 0.1*rand()/rand_max;
            p_O[0] = 1 - p_C0 - p_I0 - p_C1 - p_I1 - p_C2 - p_I2 - p_C3 - p_I3 - p_C4 - p_I4 - p_IO;
            
           
            for(int i=0; i<N-1; i++){
                // Predictor-corrector
                // Evaluate f(x, t)
                kV = 200.e-3*exp(V[i]/25.6);  
                k_V = 1.5e-3*exp(-V[i]/18.4);
                k_O = 100.e-3*exp(-V[i]/34.6); 
                dp_C0dt = -((f*f*f)*kI + 4*kV)*p_C0 + k_V*p_C1 + (k_I/(h*h*h))*p_I0;
                dp_I0dt = -(k_I/(h*h*h) + 4*kV/f)*p_I0 + h*k_V*p_I1 + (f*f*f)*kI*p_C0;
                dp_C1dt = -((f*f)*kI + 3*kV + k_V)*p_C1 + 4*kV*p_C0 + (k_I/(h*h))*p_I1 + 2*k_V*p_C2;
                dp_I1dt = -(k_I/(h*h) + h*k_V + 3*kV/f)*p_I1 + (4*kV/f)*p_I0 + (f*f)*kI*p_C1 + 2*h*k_V*p_I2;
                dp_C2dt = -(2*k_V+2*kV + f*kI)*p_C2 + (k_I/h)*p_I2 + 3*kV*p_C1 + 3*k_V*p_C3;
                dp_I2dt = -(2*h*k_V + 2*kV/f + k_I/h)*p_I2 + f*kI*p_C2 + (3*kV/f)*p_I1 + 3*h*k_V*p_I3;
                dp_C3dt = -(3*k_V + kI + kV)*p_C3 + k_I*p_I3 + 2*kV*p_C2 + 4*k_V*p_C4;
                dp_I3dt = -(3*h*k_V + kV + k_I)*p_I3 + kI*p_C3 + (2*kV/f)*p_I2 + 4*k_V*p_I4;
                dp_C4dt = -(4*k_V + kO + kI)*p_C4 + k_I*p_I4 + kV*p_C3 + k_O*p_O[i];
                dp_I4dt = -(4*k_V + kO + k_I)*p_I4 + kI*p_C4 + kV*p_I3 + k_O*p_IO;
                dp_IOdt = -(k_I + k_O)*p_IO + kI*p_O[i] + kO*p_I4;

                // Compute prediction
                pred_C0 = p_C0 + dt*dp_C0dt;
                pred_I0 = p_I0 + dt*dp_I0dt;
                pred_C1 = p_C1 + dt*dp_C1dt;
                pred_I1 = p_I1 + dt*dp_I1dt;
                pred_C2 = p_C2 + dt*dp_C2dt;
                pred_I2 = p_I2 + dt*dp_I2dt;
                pred_C3 = p_C3 + dt*dp_C3dt;
                pred_I3 = p_I3 + dt*dp_I3dt;
                pred_C4 = p_C4 + dt*dp_C4dt;
                pred_I4 = p_I4 + dt*dp_I4dt;
                pred_IO = p_IO + dt*dp_IOdt;
                pred_O = 1. - pred_C0 - pred_I0 - pred_C1 - pred_I1 - pred_C2 - pred_I2 - pred_C3 - pred_I3 - pred_C4 - pred_I4 - pred_IO;
                
                // Derivative at i+1, evaluated at prediction
                kV = 200.e-3*exp(V[i+1]/25.6);  
                k_V = 1.5e-3*exp(-V[i+1]/18.4);
                k_O = 100e-3*exp(-V[i+1]/34.6); 
                dpred_C0dt = -((f*f*f)*kI + 4*kV)*pred_C0 + k_V*pred_C1 + (k_I/(h*h*h))*pred_I0;
                dpred_I0dt = -(k_I/(h*h*h) + 4*kV/f)*pred_I0 + h*k_V*pred_I1 + (f*f*f)*kI*pred_C0;
                dpred_C1dt = -((f*f)*kI + 3*kV + k_V)*pred_C1 + 4*kV*pred_C0 + (k_I/(h*h))*pred_I1 + 2*k_V*pred_C2;
                dpred_I1dt = -(k_I/(h*h) + h*k_V + 3*kV/f)*pred_I1 + (4*kV/f)*pred_I0 + (f*f)*kI*pred_C1 + 2*h*k_V*pred_I2;
                dpred_C2dt = -(2*k_V+2*kV + f*kI)*pred_C2 + (k_I/h)*pred_I2 + 3*kV*pred_C1 + 3*k_V*pred_C3;
                dpred_I2dt = -(2*h*k_V + 2*kV/f + k_I/h)*pred_I2 + f*kI*pred_C2 + (3*kV/f)*pred_I1 + 3*h*k_V*pred_I3;
                dpred_C3dt = -(3*k_V + kI + kV)*pred_C3 + k_I*pred_I3 + 2*kV*pred_C2 + 4*k_V*pred_C4;
                dpred_I3dt = -(3*h*k_V + kV + k_I)*pred_I3 + kI*pred_C3 + (2*kV/f)*pred_I2 + 4*k_V*pred_I4;
                dpred_C4dt = -(4*k_V + kO + kI)*pred_C4 + k_I*pred_I4 + kV*pred_C3 + k_O*pred_O;
                dpred_I4dt = -(4*k_V + kO + k_I)*pred_I4 + kI*pred_C4 + kV*pred_I3 + k_O*pred_IO;
                dpred_IOdt = -(k_I + k_O)*pred_IO + kI*pred_O + kO*pred_I4;
                
                // Correction
                p_C0 = p_C0 + 0.5*dt*(dp_C0dt + dpred_C0dt);
                p_I0 = p_I0 + 0.5*dt*(dp_I0dt + dpred_I0dt);
                p_C1 = p_C1 + 0.5*dt*(dp_C1dt + dpred_C1dt);
                p_I1 = p_I1 + 0.5*dt*(dp_I1dt + dpred_I1dt);
                p_C2 = p_C2 + 0.5*dt*(dp_C2dt + dpred_C2dt);
                p_I2 = p_I2 + 0.5*dt*(dp_I2dt + dpred_I2dt); 
                p_C3 = p_C3 + 0.5*dt*(dp_C3dt + dpred_C3dt);
                p_I3 = p_I3 + 0.5*dt*(dp_I3dt + dpred_I3dt);
                p_C4 = p_C4 + 0.5*dt*(dp_C4dt + dpred_C4dt); 
                p_I4 = p_I4 + 0.5*dt*(dp_I4dt + dpred_I4dt); 
                p_IO = p_IO + 0.5*dt*(dp_IOdt + dpred_IOdt); 
                    
                p_O[i+1] = 1 - p_C0 - p_I0 - p_C1 - p_I1 - p_C2 - p_I2 - p_C3 - p_I3 - p_C4 - p_I4 - p_IO;
                               
            }
            """
        vars = ['p_dt', 'V', 'p_N', 'p_O']
        v = weave.inline(code, vars)
        return p_O

    def simulateSpikingResponse(self, I, dt):
        """
        Simulate the spiking response of the GIF-Ca model to an input current I (nA) with time step dt.
        Return a list of spike times (in ms).
        The initial conditions for the simulation is V(0)=El.
        """
        self.setDt(dt)

        (time, V, eta_sum, V_T, sps) = self.simulate(I, self.El)

        return sps

    def simulateVoltageResponse(self, I, dt):
        self.setDt(dt)

        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)

        return (spks_times, V, V_T)

    def simulate(self, I, V0):
        """
        Simulate the spiking response of the GIF-Ca model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        - V_T      : mV, firing threshold
        - spks     : ms, list of spike times
        """

        # Input parameters
        p_T = len(I)
        p_dt = self.dt

        # Model parameters
        p_gl = self.gl
        p_C = self.C
        p_El = self.El
        p_Vr = self.Vr
        p_Tref = self.Tref
        p_Vt_star = self.Vt_star
        p_DV = self.DV
        p_lambda0 = self.lambda0
        p_gCa = self.g_Ca
        p_ECa = self.E_Ca
        Tref_i = int(float(p_Tref) / p_dt)

        # Model kernels
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta = p_eta.astype('double')
        p_eta_l = len(p_eta)

        (p_gamma_support, p_gamma) = self.gamma.getInterpolatedFilter(self.dt)
        p_gamma = p_gamma.astype('double')
        p_gamma_l = len(p_gamma)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
        spks = np.array(np.zeros(p_T), dtype="double")
        eta_sum = np.array(np.zeros(p_T + 2 * p_eta_l), dtype="double")
        gamma_sum = np.array(np.zeros(p_T + 2 * p_gamma_l), dtype="double")

        # Set initial condition
        V[0] = V0

        code = """
                #include <math.h>

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);

                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);
                float Vr         = float(p_Vr);
                int   Tref_ind   = int(float(p_Tref)/dt);
                float Vt_star    = float(p_Vt_star);
                float DeltaV     = float(p_DV);
                float lambda0    = float(p_lambda0);
                float gCa        = float(p_gCa);
                float ECa        = float(p_ECa);
                int eta_l        = int(p_eta_l);
                int gamma_l      = int(p_gamma_l);
                
                float kO = 500.e-3;
                float kI = 7.e-3;
                float k_I = 0.21e-3;
                float f = 0.2;
                float h = 0.45;
                float kV, k_V, k_O;     
                float p_C0, p_I0, p_C1, p_I1, p_C2, p_I2, p_C3, p_I3, p_C4, p_I4, p_IO, p_O;       
                float dp_C0dt, dp_I0dt, dp_C1dt, dp_I1dt, dp_C2dt, dp_I2dt, dp_C3dt, dp_I3dt, dp_C4dt, dp_I4dt, dp_IOdt;       
                float dpred_C0dt, dpred_I0dt, dpred_C1dt, dpred_I1dt, dpred_C2dt, dpred_I2dt, dpred_C3dt, dpred_I3dt, dpred_C4dt, dpred_I4dt, dpred_IOdt;       
                float pred_C0, pred_I0, pred_C1, pred_I1, pred_C2, pred_I2, pred_C3, pred_I3, pred_C4, pred_I4, pred_IO, pred_O;       

                float rand_max  = float(RAND_MAX);
                float p_dontspike = 0.0 ;
                float lambda = 0.0 ;
                float r = 0.0;
                
                //Random initialization of state probabilities
                p_C0 = 0.1*rand()/rand_max;
                p_I0 = 0.1*rand()/rand_max;
                p_C1 = 0.1*rand()/rand_max;
                p_I1 = 0.1*rand()/rand_max;
                p_C2 = 0.1*rand()/rand_max;
                p_I2 = 0.1*rand()/rand_max;
                p_C3 = 0.1*rand()/rand_max;
                p_I3 = 0.1*rand()/rand_max;
                p_C4 = 0.1*rand()/rand_max;
                p_I4 = 0.1*rand()/rand_max;
                p_IO = 0.1*rand()/rand_max;
                p_O = 1 - p_C0 - p_I0 - p_C1 - p_I1 - p_C2 - p_I2 - p_C3 - p_I3 - p_C4 - p_I4 - p_IO;

                for (int t=0; t<T_ind-1; t++) {


                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] - gCa*p_O*(V[t] - ECa) );
                    
                    kV = 200.e-3*exp(V[t]/25.6);  
                    k_V = 1.5e-3*exp(-V[t]/18.4);
                    k_O = 100.e-3*exp(-V[t]/34.6); 
                    dp_C0dt = -((f*f*f)*kI + 4*kV)*p_C0 + k_V*p_C1 + (k_I/(h*h*h))*p_I0;
                    dp_I0dt = -(k_I/(h*h*h) + 4*kV/f)*p_I0 + h*k_V*p_I1 + (f*f*f)*kI*p_C0;
                    dp_C1dt = -((f*f)*kI + 3*kV + k_V)*p_C1 + 4*kV*p_C0 + (k_I/(h*h))*p_I1 + 2*k_V*p_C2;
                    dp_I1dt = -(k_I/(h*h) + h*k_V + 3*kV/f)*p_I1 + (4*kV/f)*p_I0 + (f*f)*kI*p_C1 + 2*h*k_V*p_I2;
                    dp_C2dt = -(2*k_V+2*kV + f*kI)*p_C2 + (k_I/h)*p_I2 + 3*kV*p_C1 + 3*k_V*p_C3;
                    dp_I2dt = -(2*h*k_V + 2*kV/f + k_I/h)*p_I2 + f*kI*p_C2 + (3*kV/f)*p_I1 + 3*h*k_V*p_I3;
                    dp_C3dt = -(3*k_V + kI + kV)*p_C3 + k_I*p_I3 + 2*kV*p_C2 + 4*k_V*p_C4;
                    dp_I3dt = -(3*h*k_V + kV + k_I)*p_I3 + kI*p_C3 + (2*kV/f)*p_I2 + 4*k_V*p_I4;
                    dp_C4dt = -(4*k_V + kO + kI)*p_C4 + k_I*p_I4 + kV*p_C3 + k_O*p_O;
                    dp_I4dt = -(4*k_V + kO + k_I)*p_I4 + kI*p_C4 + kV*p_I3 + k_O*p_IO;
                    dp_IOdt = -(k_I + k_O)*p_IO + kI*p_O + kO*p_I4;
    
                    // Compute prediction
                    pred_C0 = p_C0 + dt*dp_C0dt;
                    pred_I0 = p_I0 + dt*dp_I0dt;
                    pred_C1 = p_C1 + dt*dp_C1dt;
                    pred_I1 = p_I1 + dt*dp_I1dt;
                    pred_C2 = p_C2 + dt*dp_C2dt;
                    pred_I2 = p_I2 + dt*dp_I2dt;
                    pred_C3 = p_C3 + dt*dp_C3dt;
                    pred_I3 = p_I3 + dt*dp_I3dt;
                    pred_C4 = p_C4 + dt*dp_C4dt;
                    pred_I4 = p_I4 + dt*dp_I4dt;
                    pred_IO = p_IO + dt*dp_IOdt;
                    pred_O = 1. - pred_C0 - pred_I0 - pred_C1 - pred_I1 - pred_C2 - pred_I2 - pred_C3 - pred_I3 - pred_C4 - pred_I4 - pred_IO;
                    
                    // Derivative at t+1, evaluated at prediction
                    kV = 200.e-3*exp(V[t+1]/25.6);  
                    k_V = 1.5e-3*exp(-V[t+1]/18.4);
                    k_O = 100e-3*exp(-V[t+1]/34.6); 
                    dpred_C0dt = -((f*f*f)*kI + 4*kV)*pred_C0 + k_V*pred_C1 + (k_I/(h*h*h))*pred_I0;
                    dpred_I0dt = -(k_I/(h*h*h) + 4*kV/f)*pred_I0 + h*k_V*pred_I1 + (f*f*f)*kI*pred_C0;
                    dpred_C1dt = -((f*f)*kI + 3*kV + k_V)*pred_C1 + 4*kV*pred_C0 + (k_I/(h*h))*pred_I1 + 2*k_V*pred_C2;
                    dpred_I1dt = -(k_I/(h*h) + h*k_V + 3*kV/f)*pred_I1 + (4*kV/f)*pred_I0 + (f*f)*kI*pred_C1 + 2*h*k_V*pred_I2;
                    dpred_C2dt = -(2*k_V+2*kV + f*kI)*pred_C2 + (k_I/h)*pred_I2 + 3*kV*pred_C1 + 3*k_V*pred_C3;
                    dpred_I2dt = -(2*h*k_V + 2*kV/f + k_I/h)*pred_I2 + f*kI*pred_C2 + (3*kV/f)*pred_I1 + 3*h*k_V*pred_I3;
                    dpred_C3dt = -(3*k_V + kI + kV)*pred_C3 + k_I*pred_I3 + 2*kV*pred_C2 + 4*k_V*pred_C4;
                    dpred_I3dt = -(3*h*k_V + kV + k_I)*pred_I3 + kI*pred_C3 + (2*kV/f)*pred_I2 + 4*k_V*pred_I4;
                    dpred_C4dt = -(4*k_V + kO + kI)*pred_C4 + k_I*pred_I4 + kV*pred_C3 + k_O*pred_O;
                    dpred_I4dt = -(4*k_V + kO + k_I)*pred_I4 + kI*pred_C4 + kV*pred_I3 + k_O*pred_IO;
                    dpred_IOdt = -(k_I + k_O)*pred_IO + kI*pred_O + kO*pred_I4;
                    
                    // Correction
                    p_C0 = p_C0 + 0.5*dt*(dp_C0dt + dpred_C0dt);
                    p_I0 = p_I0 + 0.5*dt*(dp_I0dt + dpred_I0dt);
                    p_C1 = p_C1 + 0.5*dt*(dp_C1dt + dpred_C1dt);
                    p_I1 = p_I1 + 0.5*dt*(dp_I1dt + dpred_I1dt);
                    p_C2 = p_C2 + 0.5*dt*(dp_C2dt + dpred_C2dt);
                    p_I2 = p_I2 + 0.5*dt*(dp_I2dt + dpred_I2dt); 
                    p_C3 = p_C3 + 0.5*dt*(dp_C3dt + dpred_C3dt);
                    p_I3 = p_I3 + 0.5*dt*(dp_I3dt + dpred_I3dt);
                    p_C4 = p_C4 + 0.5*dt*(dp_C4dt + dpred_C4dt); 
                    p_I4 = p_I4 + 0.5*dt*(dp_I4dt + dpred_I4dt); 
                    p_IO = p_IO + 0.5*dt*(dp_IOdt + dpred_IOdt); 
                        
                    p_O = 1 - p_C0 - p_I0 - p_C1 - p_I1 - p_C2 - p_I2 - p_C3 - p_I3 - p_C4 - p_I4 - p_IO;

                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t+1]-Vt_star-gamma_sum[t])/DeltaV );
                    p_dontspike = exp(-lambda*(dt/1000.0));    

                    // PRODUCE SPIKE STOCHASTICALLY
                    r = rand()/rand_max;
                    if (r > p_dontspike) {

                        if (t+1 < T_ind-1)
                            spks[t+1] = 1.0;

                        t = t + Tref_ind;

                        if (t+1 < T_ind-1)
                            V[t+1] = Vr;

                        // UPDATE ADAPTATION PROCESSES
                        for(int j=0; j<eta_l; j++)
                            eta_sum[t+1+j] += p_eta[j];

                        for(int j=0; j<gamma_l; j++)
                            gamma_sum[t+1+j] += p_gamma[j] ;

                    }

                }
                """

        vars = ['p_T', 'p_dt', 'p_gl', 'p_C', 'p_El', 'p_Vr', 'p_Tref', 'p_Vt_star', 'p_DV', 'p_lambda0', 'p_gCa',
                'p_ECa', 'V', 'I',
                'p_eta', 'p_eta_l', 'eta_sum', 'p_gamma', 'gamma_sum', 'p_gamma_l', 'spks']

        v = weave.inline(code, vars)

        time = np.arange(p_T) * self.dt

        eta_sum = eta_sum[:p_T]
        V_T = gamma_sum[:p_T] + p_Vt_star

        spks = (np.where(spks == 1)[0]) * self.dt

        return (time, V, eta_sum, V_T, spks)

    def simulateDeterministic_forceSpikes(self, I, V0, spks):
        """
        Simulate the subthresohld response of the GIF-Ca model to an input current I (nA) with time step dt.
        Adaptation currents are enforced at times specified in the list spks (in ms) given as an argument to the function.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        """
        # Input parameters
        p_T = len(I)
        p_dt = self.dt

        # Model parameters
        p_gl = self.gl
        p_C = self.C
        p_El = self.El
        p_Vr = self.Vr
        p_Tref = self.Tref
        p_Tref_i = int(self.Tref / self.dt)
        p_gCa = self.g_Ca
        p_ECa = self.E_Ca

        # Model kernel
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta = p_eta.astype('double')
        p_eta_l = len(p_eta)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
        I_Ca = np.array(np.zeros(p_T), dtype="double")

        spks = np.array(spks, dtype="double")
        spks_i = Tools.timeToIndex(spks, self.dt)

        # Compute adaptation current (sum of eta triggered at spike times in spks)
        eta_sum = np.array(np.zeros(int(p_T + 1.1 * p_eta_l + p_Tref_i)), dtype="double")

        for s in spks_i:
            eta_sum[s + 1 + p_Tref_i: s + 1 + p_Tref_i + p_eta_l] += p_eta

        eta_sum = eta_sum[:p_T]

        # Set initial condition
        V[0] = V0

        code = """
                #include <math.h>

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);
                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);
                float Vr         = float(p_Vr);
                int   Tref_ind   = int(float(p_Tref)/dt);
                float gCa        = float(p_gCa);
                float ECa        = float(p_ECa);

                int next_spike = spks_i[0] + Tref_ind;
                int spks_cnt = 0;
 
                float kO = 500.e-3;
                float kI = 7.e-3;
                float k_I = 0.21e-3;
                float f = 0.2;
                float h = 0.45;
                float kV, k_V, k_O;     
                float p_C0, p_I0, p_C1, p_I1, p_C2, p_I2, p_C3, p_I3, p_C4, p_I4, p_IO, p_O;       
                float dp_C0dt, dp_I0dt, dp_C1dt, dp_I1dt, dp_C2dt, dp_I2dt, dp_C3dt, dp_I3dt, dp_C4dt, dp_I4dt, dp_IOdt;       
                float dpred_C0dt, dpred_I0dt, dpred_C1dt, dpred_I1dt, dpred_C2dt, dpred_I2dt, dpred_C3dt, dpred_I3dt, dpred_C4dt, dpred_I4dt, dpred_IOdt;       
                float pred_C0, pred_I0, pred_C1, pred_I1, pred_C2, pred_I2, pred_C3, pred_I3, pred_C4, pred_I4, pred_IO, pred_O;       

                float rand_max  = float(RAND_MAX);
                  
                //Random initialization of state probabilities
                p_C0 = 0.1*rand()/rand_max;
                p_I0 = 0.1*rand()/rand_max;
                p_C1 = 0.1*rand()/rand_max;
                p_I1 = 0.1*rand()/rand_max;
                p_C2 = 0.1*rand()/rand_max;
                p_I2 = 0.1*rand()/rand_max;
                p_C3 = 0.1*rand()/rand_max;
                p_I3 = 0.1*rand()/rand_max;
                p_C4 = 0.1*rand()/rand_max;
                p_I4 = 0.1*rand()/rand_max;
                p_IO = 0.1*rand()/rand_max;
                p_O = 1 - p_C0 - p_I0 - p_C1 - p_I1 - p_C2 - p_I2 - p_C3 - p_I3 - p_C4 - p_I4 - p_IO;
                
                
                for (int t=0; t<T_ind-1; t++) {

                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] - gCa*p_O*(V[t] - ECa) );
                    kV = 200.e-3*exp(V[t]/25.6);  
                    k_V = 1.5e-3*exp(-V[t]/18.4);
                    k_O = 100.e-3*exp(-V[t]/34.6); 
                    dp_C0dt = -((f*f*f)*kI + 4*kV)*p_C0 + k_V*p_C1 + (k_I/(h*h*h))*p_I0;
                    dp_I0dt = -(k_I/(h*h*h) + 4*kV/f)*p_I0 + h*k_V*p_I1 + (f*f*f)*kI*p_C0;
                    dp_C1dt = -((f*f)*kI + 3*kV + k_V)*p_C1 + 4*kV*p_C0 + (k_I/(h*h))*p_I1 + 2*k_V*p_C2;
                    dp_I1dt = -(k_I/(h*h) + h*k_V + 3*kV/f)*p_I1 + (4*kV/f)*p_I0 + (f*f)*kI*p_C1 + 2*h*k_V*p_I2;
                    dp_C2dt = -(2*k_V+2*kV + f*kI)*p_C2 + (k_I/h)*p_I2 + 3*kV*p_C1 + 3*k_V*p_C3;
                    dp_I2dt = -(2*h*k_V + 2*kV/f + k_I/h)*p_I2 + f*kI*p_C2 + (3*kV/f)*p_I1 + 3*h*k_V*p_I3;
                    dp_C3dt = -(3*k_V + kI + kV)*p_C3 + k_I*p_I3 + 2*kV*p_C2 + 4*k_V*p_C4;
                    dp_I3dt = -(3*h*k_V + kV + k_I)*p_I3 + kI*p_C3 + (2*kV/f)*p_I2 + 4*k_V*p_I4;
                    dp_C4dt = -(4*k_V + kO + kI)*p_C4 + k_I*p_I4 + kV*p_C3 + k_O*p_O;
                    dp_I4dt = -(4*k_V + kO + k_I)*p_I4 + kI*p_C4 + kV*p_I3 + k_O*p_IO;
                    dp_IOdt = -(k_I + k_O)*p_IO + kI*p_O + kO*p_I4;
    
                    // Compute prediction
                    pred_C0 = p_C0 + dt*dp_C0dt;
                    pred_I0 = p_I0 + dt*dp_I0dt;
                    pred_C1 = p_C1 + dt*dp_C1dt;
                    pred_I1 = p_I1 + dt*dp_I1dt;
                    pred_C2 = p_C2 + dt*dp_C2dt;
                    pred_I2 = p_I2 + dt*dp_I2dt;
                    pred_C3 = p_C3 + dt*dp_C3dt;
                    pred_I3 = p_I3 + dt*dp_I3dt;
                    pred_C4 = p_C4 + dt*dp_C4dt;
                    pred_I4 = p_I4 + dt*dp_I4dt;
                    pred_IO = p_IO + dt*dp_IOdt;
                    pred_O = 1. - pred_C0 - pred_I0 - pred_C1 - pred_I1 - pred_C2 - pred_I2 - pred_C3 - pred_I3 - pred_C4 - pred_I4 - pred_IO;
                    
                    // Derivative at t+1, evaluated at prediction
                    kV = 200.e-3*exp(V[t+1]/25.6);  
                    k_V = 1.5e-3*exp(-V[t+1]/18.4);
                    k_O = 100e-3*exp(-V[t+1]/34.6); 
                    dpred_C0dt = -((f*f*f)*kI + 4*kV)*pred_C0 + k_V*pred_C1 + (k_I/(h*h*h))*pred_I0;
                    dpred_I0dt = -(k_I/(h*h*h) + 4*kV/f)*pred_I0 + h*k_V*pred_I1 + (f*f*f)*kI*pred_C0;
                    dpred_C1dt = -((f*f)*kI + 3*kV + k_V)*pred_C1 + 4*kV*pred_C0 + (k_I/(h*h))*pred_I1 + 2*k_V*pred_C2;
                    dpred_I1dt = -(k_I/(h*h) + h*k_V + 3*kV/f)*pred_I1 + (4*kV/f)*pred_I0 + (f*f)*kI*pred_C1 + 2*h*k_V*pred_I2;
                    dpred_C2dt = -(2*k_V+2*kV + f*kI)*pred_C2 + (k_I/h)*pred_I2 + 3*kV*pred_C1 + 3*k_V*pred_C3;
                    dpred_I2dt = -(2*h*k_V + 2*kV/f + k_I/h)*pred_I2 + f*kI*pred_C2 + (3*kV/f)*pred_I1 + 3*h*k_V*pred_I3;
                    dpred_C3dt = -(3*k_V + kI + kV)*pred_C3 + k_I*pred_I3 + 2*kV*pred_C2 + 4*k_V*pred_C4;
                    dpred_I3dt = -(3*h*k_V + kV + k_I)*pred_I3 + kI*pred_C3 + (2*kV/f)*pred_I2 + 4*k_V*pred_I4;
                    dpred_C4dt = -(4*k_V + kO + kI)*pred_C4 + k_I*pred_I4 + kV*pred_C3 + k_O*pred_O;
                    dpred_I4dt = -(4*k_V + kO + k_I)*pred_I4 + kI*pred_C4 + kV*pred_I3 + k_O*pred_IO;
                    dpred_IOdt = -(k_I + k_O)*pred_IO + kI*pred_O + kO*pred_I4;
                    
                    // Correction
                    p_C0 = p_C0 + 0.5*dt*(dp_C0dt + dpred_C0dt);
                    p_I0 = p_I0 + 0.5*dt*(dp_I0dt + dpred_I0dt);
                    p_C1 = p_C1 + 0.5*dt*(dp_C1dt + dpred_C1dt);
                    p_I1 = p_I1 + 0.5*dt*(dp_I1dt + dpred_I1dt);
                    p_C2 = p_C2 + 0.5*dt*(dp_C2dt + dpred_C2dt);
                    p_I2 = p_I2 + 0.5*dt*(dp_I2dt + dpred_I2dt); 
                    p_C3 = p_C3 + 0.5*dt*(dp_C3dt + dpred_C3dt);
                    p_I3 = p_I3 + 0.5*dt*(dp_I3dt + dpred_I3dt);
                    p_C4 = p_C4 + 0.5*dt*(dp_C4dt + dpred_C4dt); 
                    p_I4 = p_I4 + 0.5*dt*(dp_I4dt + dpred_I4dt); 
                    p_IO = p_IO + 0.5*dt*(dp_IOdt + dpred_IOdt); 
                        
                    p_O = 1 - p_C0 - p_I0 - p_C1 - p_I1 - p_C2 - p_I2 - p_C3 - p_I3 - p_C4 - p_I4 - p_IO;                 
                    //I_Ca[t+1] = -gCa*p_O*(V[t+1] - ECa);
                    if ( t == next_spike ) {
                        spks_cnt = spks_cnt + 1;
                        next_spike = spks_i[spks_cnt] + Tref_ind;
                        V[t-1] = 0 ;
                        V[t] = Vr ;
                        t=t-1;
                    }

                }
                """

        vars = ['p_T', 'p_dt', 'p_gl', 'p_C', 'p_El', 'p_Vr', 'p_Tref', 'p_gCa', 'p_ECa', 'V', 'I',
                'I_Ca', 'eta_sum', 'spks_i']

        v = weave.inline(code, vars)

        time = np.arange(p_T) * self.dt

        return time, V, eta_sum



    ########################################################################################################
    # FUNCTIONS FOR FITTING
    ########################################################################################################

    def fit(self, experiment, DT_beforeSpike=5.0, is_E_Ca_fixed=False):

        """
        Fit the GIF-Ca model on experimental data.
        The experimental data are stored in the object experiment.
        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        """

        # Three step procedure used for parameters extraction

        print "\n################################"
        print "# Fit GIF-Ca-Kinetic model"
        print "################################\n"

        self.fitVoltageReset(experiment, self.Tref, do_plot=False)

        (var_explained_dV, var_explained_V) = self.fitSubthresholdDynamics(experiment, is_E_Ca_fixed,
                                                                           DT_beforeSpike=DT_beforeSpike)

        self.fitStaticThreshold(experiment)

        self.fitThresholdDynamics(experiment)
        return (var_explained_dV, var_explained_V)

    ########################################################################################################
    # FUNCTIONS RELATED TO FIT OF SUBTHRESHOLD DYNAMICS (step 2)
    ########################################################################################################

    def fitSubthresholdDynamics(self, experiment, is_E_Ca_fixed, DT_beforeSpike=5.0):

        print "\nGIF-Ca-Kinetic MODEL - Fit subthreshold dynamics..."

        # Expand eta in basis functions
        self.dt = experiment.dt
        self.eta.computeBins()

        # Build X matrix and Y vector to perform linear regression (use all traces in training set)
        X = []
        Y = []

        cnt = 0

        for tr in experiment.trainingset_traces:

            if tr.useTrace:
                cnt += 1
                reprint("Compute X matrix for repetition %d" % (cnt))

                (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, is_E_Ca_fixed,
                                                                                    DT_beforeSpike=DT_beforeSpike)

                X.append(X_tmp)
                Y.append(Y_tmp)

        # Concatenate matrixes associated with different traces to perform a single multilinear regression
        if cnt == 1:
            X = X[0]
            Y = Y[0]

        elif cnt > 1:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

        else:
            print "\nError, at least one training set trace should be selected to perform fit."

        # Linear Regression
        print "\nPerform linear regression..."
        XTX = np.dot(np.transpose(X), X)
        XTX_inv = inv(XTX)
        XTY = np.dot(np.transpose(X), Y)
        b = np.dot(XTX_inv, XTY)
        b = b.flatten()

        # Update and print model parameters
        self.C = 1. / b[1]
        self.gl = -b[0] * self.C
        self.El = b[2] * self.C / self.gl

        if not is_E_Ca_fixed:
            self.g_Ca = -b[-2] * self.C
            self.E_Ca = b[-1] * self.C / self.g_Ca
            self.eta.setFilter_Coefficients(-b[3:-2] * self.C)
        else:
            self.g_Ca = -b[-1] * self.C
            self.eta.setFilter_Coefficients(-b[3:-1] * self.C)

        self.printParameters()

        # Compute percentage of variance explained on dV/dt

        var_explained_dV = 1.0 - np.mean((Y - np.dot(X, b)) ** 2) / np.var(Y)
        print "Percentage of variance explained (on dV/dt) : %0.2f" % (var_explained_dV * 100.0)

        # Compute percentage of variance explained on V

        SSE = 0  # sum of squared errors
        VAR = 0  # variance of data

        for tr in experiment.trainingset_traces:

            if tr.useTrace:
                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                indices_tmp = tr.getROI_FarFromSpikes(DT_beforeSpike, self.Tref)

                SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp]) ** 2)
                VAR += len(indices_tmp) * np.var(tr.V[indices_tmp])

        var_explained_V = 1.0 - SSE / VAR

        print "Percentage of variance explained (on V) : %0.2f" % (var_explained_V * 100.0)
        return (var_explained_dV * 100.0, var_explained_V * 100.0)

    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace, is_E_Ca_fixed, DT_beforeSpike=5.0):

        # Length of the voltage trace
        Tref_ind = int(self.Tref / trace.dt)

        # Select region where to perform linear regression
        selection = trace.getROI_FarFromSpikes(DT_beforeSpike, self.Tref)
        selection_l = len(selection)

        # Build X matrix for linear regression
        X = np.zeros((selection_l, 3))

        # Fill first two columns of X matrix
        X[:, 0] = trace.V[selection]
        X[:, 1] = trace.I[selection]
        X[:, 2] = np.ones(selection_l)

        # Compute and fill the columns associated with the spike-triggered current eta
        X_eta = self.eta.convolution_Spiketrain_basisfunctions(trace.getSpikeTimes() + self.Tref, trace.T, trace.dt)
        X = np.concatenate((X, X_eta[selection, :]), axis=1)

        # Compute and fill columns associated with the calcium current
        p_O = self.simulate_openprob(trace.V)
        p_O = p_O[selection]

        if not is_E_Ca_fixed:
            tmp = p_O * X[:, 0]
            X = np.concatenate((X, tmp.reshape((selection_l, 1))), axis=1)
            tmp = p_O
            X = np.concatenate((X, tmp.reshape((selection_l, 1))), axis=1)
        else:
            tmp = p_O * (X[:, 0] - self.E_Ca)
            X = np.concatenate((X, tmp.reshape((selection_l, 1))), axis=1)

        # Build Y vector (voltage derivative)

        # COULD BE A BETTER SOLUTION IN CASE OF EXPERIMENTAL DATA (NOT CLEAR WHY)
        # Y = np.array( np.concatenate( ([0], np.diff(trace.V)/trace.dt) ) )[selection]

        # Better approximation for the derivative (modification by AP, september 2017)
        Y = np.gradient(trace.V, trace.dt)[selection]

        # CORRECT SOLUTION TO FIT ARTIFICIAL DATA
        # Y = np.array( np.concatenate( (np.diff(trace.V)/trace.dt, [0]) ) )[selection]
        return (X, Y)

    ##############################################################################################################
    # PRINT PARAMETRES
    ##############################################################################################################

    def printParameters(self):

        print "\n-------------------------"
        print "GIF-Ca-Kinetic model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t%0.3f" % (self.C / self.gl)
        print "R (MOhm):\t%0.9f" % (1.0 / self.gl)
        print "C (nF):\t\t%0.3f" % (self.C)
        print "gl (uS):\t%0.3f" % (self.gl)
        print "El (mV):\t%0.3f" % (self.El)
        print "Tref (ms):\t%0.3f" % (self.Tref)
        print "Vr (mV):\t%0.3f" % (self.Vr)
        print "Vt* (mV):\t%0.3f" % (self.Vt_star)
        print "DV (mV):\t%0.3f" % (self.DV)
        print "g_Ca (uS):\t%0.3f" % (self.g_Ca)
        print "ECa (mV):\t%0.3f" % (self.E_Ca)
        print "-------------------------\n"

