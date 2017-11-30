import matplotlib.pyplot as plt
import numpy as np

import weave
from weave import converters
from numpy.linalg import inv

from SpikingModel import *
from iGIF import *

from Filter_Rect_LogSpaced import *

import Tools
from Tools import reprint

class GIF_Ca(GIF) :
    """
     Generalized Integrate and Fire model with voltage-dependent calcium current (VDCC).
     Spike are produced stochastically with firing intensity:
     lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),
     where the membrane potential dynamics is given by:

     C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j) - g_Ca*m*h(V-E_Ca),

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

        self.E_Ca = 20.0  # mV, reversal potential associated with the voltage-dependent calcium current
        self.g_Ca = 0.01  # uS, maximal conductance of VDCC
        self.shift = 0. # mV shift in the in/activation curves with respect to some ground truth

    def set_shift(self, val):
        self.shift = val

    @staticmethod
    def m_inf(V):
        return 1./( (1. + np.exp((-60.2 - V)/22.4)) )

    @staticmethod
    def h_inf(V):
        return 1./(1. + np.exp((V+63.)/8.15))

    @staticmethod
    def tau_m(V):
        return 2.4 + 22.5/np.cosh((V+39.)/12.)

    @staticmethod
    def tau_h(V):
        return (V <= -50.)*(127. + 0.21*np.exp(50./6.5)) + (V > -50.)*(127. + 0.21*np.exp(-V/6.5))


    ########################################################################################################
    # FUNCTIONS FOR SIMULATIONS
    ########################################################################################################

    def simulate_h(self, V):
        """
        :param V: vector of voltages for a given training
        :return: array of h
        """

        h = np.array(np.zeros(V.size), dtype="double")
        h[0] = self.h_inf(V[0]-self.shift)
        p_dt = self.dt
        p_N = V.size
        V = np.array(V, dtype="double")
        h_inf_vec = np.array(self.h_inf(V-self.shift), dtype="double")
        tau_h_vec = np.array(self.tau_h(V-self.shift), dtype="double")

        code = """
            #include <math.h>
            float dt = float(p_dt);
            int N    = int(p_N);
            double h_inf;
            double tau_h;

            for(int i=0; i<N-1; i++){
                h[i+1] = h[i] + (dt/tau_h_vec[i])*(h_inf_vec[i] - h[i]);               
            }
            """
        vars = ['h', 'p_dt', 'V', 'p_N', 'h_inf_vec', 'tau_h_vec']
        v = weave.inline(code, vars)
        return h

    def simulate_m(self, V):
        """
        :param V: vector of voltages for a given training
        :return: array of m
        """
        m = np.array(np.zeros(V.size), dtype="double")
        m[0] = self.m_inf(V[0]-self.shift)
        p_dt = self.dt
        p_N = V.size
        V = np.array(V, dtype="double")
        m_inf_vec = np.array(self.m_inf(V-self.shift), dtype="double")
        tau_m_vec = np.array(self.tau_m(V-self.shift), dtype="double")

        code = """
              #include <math.h>
              float dt = float(p_dt);
              int N    = int(p_N);
              double m_inf;
              double tau_m;
 
              for(int i=0; i<N-1; i++){
                  m[i+1] = ( m[i] + (dt/tau_m_vec[i])*m_inf_vec[i] )/(1 + dt/tau_m_vec[i]);
              }
              """
        vars = ['m', 'p_dt', 'V', 'p_N', 'm_inf_vec', 'tau_m_vec']
        v = weave.inline(code, vars)
        return m


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
        p_shift = self.shift
        Tref_i = int(float(p_Tref)/p_dt)

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
        """
        m = GIF_Ca.m_inf(V0-p_shift)
        h = GIF_Ca.h_inf(V0-p_shift)

        for t in xrange(p_T-1):
            # INTEGRATE VOLTAGE
            V[t + 1] = V[t] + p_dt / p_C * (-p_gl * (V[t] - p_El) + I[t] - eta_sum[t] - p_gCa * m * m * m * m * h*(V[t] - p_ECa))
            m = m + (p_dt / GIF_Ca.tau_m(V[t]-p_shift)) * (GIF_Ca.m_inf(V[t]-p_shift) - m)
            h = h + (p_dt / GIF_Ca.tau_h(V[t]-p_shift)) * (GIF_Ca.h_inf(V[t]-p_shift) - h)
            # COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
            lambd = p_lambda0 * np.exp( (V[t+1]-p_Vt_star-gamma_sum[t]) / p_DV )
            p_dontspike = np.exp(-lambd * (p_dt / 1000.0))
            #PRODUCE SPIKE STOCHASTICALLY
            r = np.random.rand()
            if r > p_dontspike:
                if t + 1 < p_T - 1:
                    spks[t + 1] = 1.0
                    V[t + 1] = p_Vr

                t = t + Tref_i

                # UPDATE ADAPTATION PROCESSES
                for j in xrange(p_eta_l):
                    eta_sum[t + 1 + j] += p_eta[j]

                for j in xrange(p_gamma_l):
                    gamma_sum[t + 1 + j] += p_gamma[j]
        """

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
                float shift      = float(p_shift);
                int eta_l        = int(p_eta_l);
                int gamma_l      = int(p_gamma_l);

                float rand_max  = float(RAND_MAX);
                float p_dontspike = 0.0 ;
                float lambda = 0.0 ;
                float r = 0.0;
                float m = 1./(1. + exp((-60.2 - V[0]+shift)/22.4));
                float h = 1. / (1. + exp((V[0] - shift + 63.) / 8.15));
                float m_inf_val;
                float h_inf_val;
                float tau_m_val;
                float tau_h_val;
                
                for (int t=0; t<T_ind-1; t++) {


                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] - gCa*m*m*m*m*h*(V[t] - ECa) );
                    m_inf_val = 1./( (1. + exp((-60.2 - (V[t]-shift))/22.4)) );
                    h_inf_val = 1. / (1. + exp((V[t] - shift + 63.) / 8.15));
                    tau_m_val = 2.4 + 22.5/cosh((V[t]-shift+39.)/12.);
                    tau_h_val = ((V[t]-shift) <= -50.)*(127. + 0.21*exp(50./6.5)) + ((V[t]-shift) > -50.)*(127. + 0.21*exp(-((V[t]-shift))/6.5));

                    m = m + (dt/tau_m_val)*( m_inf_val - m );
                    h = h + (dt/tau_h_val)*( h_inf_val - h );

                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t+1]-Vt_star-gamma_sum[t])/DeltaV );
                    p_dontspike = exp(-lambda*(dt/1000.0));                                  // since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)


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
                'p_ECa', 'p_shift', 'V', 'I',
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
        p_shift = self.shift

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
        """
        m = GIF_Ca.m_inf(V0)
        h = GIF_Ca.h_inf(V0)
        I_Ca[0] = -p_gCa * m * m * m * m * h * (V0 - p_ECa)
        spks_cnt = 0
        next_spike = spks_i[0] + p_Tref_i

        for t in xrange(p_T - 1):
            # INTEGRATE VOLTAGE
            V[t + 1] = V[t] + p_dt / p_C * (
            -p_gl * (V[t] - p_El) + I[t] - eta_sum[t] - p_gCa * m * m * m * m * h * (V[t] - p_ECa))
            m = m + (p_dt / GIF_Ca.tau_m(V[t] - p_shift)) * (GIF_Ca.m_inf(V[t] - p_shift) - m)
            h = h + (p_dt / GIF_Ca.tau_h(V[t] - p_shift)) * (GIF_Ca.h_inf(V[t] - p_shift) - h)
            I_Ca[t + 1] = -p_gCa * m * m * m * m * h * (V[t + 1] - p_ECa)
            if t == next_spike :
                spks_cnt = spks_cnt + 1
                if spks_cnt < len(spks_i):
                    next_spike = spks_i[spks_cnt] + p_Tref_i
                V[t-1] = 0
                V[t] = p_Vr
                t=t-1



        """

        code =  """
                #include <math.h>

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);
                float shift      = float(p_shift);
                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);
                float Vr         = float(p_Vr);
                int   Tref_ind   = int(float(p_Tref)/dt);
                float gCa        = float(p_gCa);
                float ECa        = float(p_ECa);

                int next_spike = spks_i[0] + Tref_ind;
                int spks_cnt = 0;
                float m = 1./(1. + exp((-60.2 - V[0]+shift)/22.4));
                float h = 1. / (1. + exp((V[0] - shift + 63.) / 8.15));

                //I_Ca[0] = -gCa*m*m*m*m*h*(V[0] - ECa);
                
                float m_inf_val;
                float h_inf_val;
                float tau_m_val;
                float tau_h_val;

                for (int t=0; t<T_ind-1; t++) {


                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] - gCa*m*m*m*m*h*(V[t] - ECa) );
                    m_inf_val = 1./( (1. + exp((-60.2 - (V[t]-shift))/22.4)) );
                    h_inf_val = 1. / (1. + exp((V[t] - shift + 63.) / 8.15));
                    tau_m_val = 2.4 + 22.5/cosh((V[t]-shift+39.)/12.);
                    tau_h_val = ((V[t]-shift) <= -50.)*(127. + 0.21*exp(50./6.5)) + ((V[t]-shift) > -50.)*(127. + 0.21*exp(-((V[t]-shift))/6.5));

                    m = m + (dt/tau_m_val)*( m_inf_val - m );
                    h = h + (dt/tau_h_val)*( h_inf_val - h );
                    //I_Ca[t+1] = -gCa*m*m*m*m*h*(V[t+1] - ECa);

                    if ( t == next_spike ) {
                        spks_cnt = spks_cnt + 1;
                        next_spike = spks_i[spks_cnt] + Tref_ind;
                        V[t-1] = 0 ;
                        V[t] = Vr ;
                        t=t-1;
                    }

                }
                """

        vars = ['p_T', 'p_dt', 'p_gl', 'p_C', 'p_El', 'p_Vr', 'p_Tref', 'p_shift', 'p_gCa', 'p_ECa', 'V', 'I',
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
        print "# Fit GIF-Ca model"
        print "################################\n"

        self.fitVoltageReset(experiment, self.Tref, do_plot=False)

        (var_explained_dV, var_explained_V) = self.fitSubthresholdDynamics(experiment, is_E_Ca_fixed, DT_beforeSpike=DT_beforeSpike)

        self.fitStaticThreshold(experiment)

        self.fitThresholdDynamics(experiment)
        return (var_explained_dV, var_explained_V)

    ########################################################################################################
    # FUNCTIONS RELATED TO FIT OF SUBTHRESHOLD DYNAMICS (step 2)
    ########################################################################################################

    def fitSubthresholdDynamics(self, experiment, is_E_Ca_fixed, DT_beforeSpike=5.0):

        print "\nGIF-Ca MODEL - Fit subthreshold dynamics..."

        # Expand eta in basis functions
        self.dt = experiment.dt
        self.eta.computeBins()

        # Build X matrix and Y vector to perform linear regression (use all traces in training set)
        X = []
        Y = []

        cnt = 0

        for tr in experiment.trainingset_traces :

            if tr.useTrace :

                cnt += 1
                reprint( "Compute X matrix for repetition %d" % (cnt) )

                (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, is_E_Ca_fixed, DT_beforeSpike=DT_beforeSpike)

                X.append(X_tmp)
                Y.append(Y_tmp)


        # Concatenate matrixes associated with different traces to perform a single multilinear regression
        if cnt == 1:
            X = X[0]
            Y = Y[0]

        elif cnt > 1:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

        else :
            print "\nError, at least one training set trace should be selected to perform fit."


        # Linear Regression
        print "\nPerform linear regression..."
        XTX     = np.dot(np.transpose(X), X)
        XTX_inv = inv(XTX)
        XTY     = np.dot(np.transpose(X), Y)
        b       = np.dot(XTX_inv, XTY)
        b       = b.flatten()


        # Update and print model parameters
        self.C = 1. / b[1]
        self.gl = -b[0] * self.C
        self.El = b[2] * self.C / self.gl

        if not is_E_Ca_fixed:
            self.g_Ca = -b[-2]*self.C
            self.E_Ca = b[-1]*self.C/self.g_Ca
            self.eta.setFilter_Coefficients(-b[3:-2]*self.C)
        else:
            self.g_Ca = -b[-1] * self.C
            self.eta.setFilter_Coefficients(-b[3:-1] * self.C)

        self.printParameters()


        # Compute percentage of variance explained on dV/dt

        var_explained_dV = 1.0 - np.mean((Y - np.dot(X,b))**2)/np.var(Y)
        print "Percentage of variance explained (on dV/dt) for shift %0.1f: %0.2f" % (self.shift, var_explained_dV*100.0)


        # Compute percentage of variance explained on V

        SSE = 0     # sum of squared errors
        VAR = 0     # variance of data

        for tr in experiment.trainingset_traces :

            if tr.useTrace :

                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                indices_tmp = tr.getROI_FarFromSpikes(DT_beforeSpike, self.Tref)

                SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])

        var_explained_V = 1.0 - SSE / VAR

        print "Percentage of variance explained (on V) for shift %0.1f: %0.2f" % (self.shift, var_explained_V*100.0)
        return (var_explained_dV*100.0, var_explained_V*100.0)

    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace, is_E_Ca_fixed, DT_beforeSpike=5.0):

        # Length of the voltage trace
        Tref_ind = int(self.Tref/trace.dt)

        # Select region where to perform linear regression
        selection = trace.getROI_FarFromSpikes(DT_beforeSpike, self.Tref)
        selection_l = len(selection)

        # Build X matrix for linear regression
        X = np.zeros( (selection_l, 3) )

        # Fill first two columns of X matrix
        X[:,0] = trace.V[selection]
        X[:,1] = trace.I[selection]
        X[:,2] = np.ones(selection_l)


        # Compute and fill the columns associated with the spike-triggered current eta
        X_eta = self.eta.convolution_Spiketrain_basisfunctions(trace.getSpikeTimes() + self.Tref, trace.T, trace.dt)
        X = np.concatenate( (X, X_eta[selection,:]), axis=1 )

        #Compute and fill columns associated with the calcium current
        m = self.simulate_m(trace.V)
        m = m[selection]
        h = self.simulate_h(trace.V)
        h = h[selection]

        if not is_E_Ca_fixed:
            tmp = m**4*h*X[:,0]
            X = np.concatenate((X, tmp.reshape((selection_l,1))), axis=1)
            tmp = m**4*h
            X = np.concatenate((X, tmp.reshape((selection_l,1))), axis=1)
        else:
            tmp = m**4 * h * (X[:, 0] - self.E_Ca)
            X = np.concatenate((X, tmp.reshape((selection_l, 1))), axis=1)

        # Build Y vector (voltage derivative)

        # COULD BE A BETTER SOLUTION IN CASE OF EXPERIMENTAL DATA (NOT CLEAR WHY)
        #Y = np.array( np.concatenate( ([0], np.diff(trace.V)/trace.dt) ) )[selection]

        #Better approximation for the derivative (modification by AP, september 2017)
        Y = np.gradient(trace.V, trace.dt)[selection]

        # CORRECT SOLUTION TO FIT ARTIFICIAL DATA
        #Y = np.array( np.concatenate( (np.diff(trace.V)/trace.dt, [0]) ) )[selection]
        return (X, Y)



    ##############################################################################################################
    # PRINT PARAMETRES
    ##############################################################################################################

    def printParameters(self):

        print "\n-------------------------"
        print "GIF-Ca model parameters:"
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

