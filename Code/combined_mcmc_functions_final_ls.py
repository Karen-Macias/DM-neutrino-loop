import scipy
import mpmath
import numpy as np
from scipy.integrate import quad
from scipy.misc import derivative

#constants
gx = 1                  #Real scalar DM coupling
mw = 80.377             #W boson mass in GeV, PDG 2022 fit
mz = 91.1876            #Z boson mass in GeV, PDG 2023 fit
Tnu = 1.68e-13          #CNB temperature today in GeV
m_K = 0.493677          #Kaon mass in GeV, PDG 2022
m_D = 1.86966           #D meson mass in GeV, PDG 2022
al = 7.2973525693e-3    #fine structure constant, PDG 2022
sinw = np.sqrt(0.23121) #weak-mixing angle, PDG 2022
cosw = np.sqrt(1 - 0.23121)
h_bar = 6.582119569e-25 #GeV s
c = 2.99792458e+10 #cgs
cgs_to_gev = 1/(h_bar**2*c**3)

def DiLog(z,s):
    dilog = np.frompyfunc(mpmath.polylog,2,1)
    if s<0:
        ans = dilog(2,z)
    else:
        ans = -(1/2)*mpmath.log(1/(1-z))**2-dilog(2,z/(-1+z))
    return ans

@mpmath.workdps(50)
def squared_amplitude_s(s,mphi1,mphi2,mN,lam,Ynu):
    def C0(s,mphi,mN):
        polylog = np.frompyfunc(mpmath.polylog,2,1)
        sqrt = np.frompyfunc(mpmath.sqrt,1,1)
        alpha = mphi**2/mN**2
        beta = s/mN**2
        a = 2*(1-alpha)/(2-2*alpha+beta-sqrt(beta*(beta-4*alpha)))
        b = (2*(1-alpha+beta))/(2-2*alpha+beta-sqrt(beta*(beta-4*alpha))) #typo (2*(1-alpha)+beta) -> (2*(1-alpha+beta))
        c = (2*(1-alpha))/(2-2*alpha+beta+sqrt(beta*(beta-4*alpha)))
    
        if a.imag!=0:          #it's a complex number
            one = polylog(2,a)
        elif a.real<=1:
            one = polylog(2,a)
        else:
            one = DiLog(a,s)
    
        if b.imag!=0:          #it's a complex number
            two = polylog(2,b)
        elif b.real<=1:
            two = polylog(2,b)
        else:
            two = DiLog(b,s)
    
        if c.imag!=0:          #it's a complex number
            three = polylog(2,c)
        elif c.real<=1:
            three = polylog(2,c)
        else:
            three = DiLog(c,s)
        
        d = (1-alpha)**2/(1-2*alpha+alpha**2+beta)
        e = ((1-alpha)*(1-alpha+beta))/(1-2*alpha+alpha**2+beta)
        f = (2*(1-alpha+beta))/(2-2*alpha+beta+sqrt(beta*(beta-4*alpha)))
        
        four=polylog(2,d)
        five=polylog(2,e)
        six=polylog(2,f)
       
        amplitude = (1/s)*(one-two+three-four+five-six)
        
        #approximation for low momentum transfer s-->0
        if amplitude.real==0:
        #if beta<1e-13:
            diff = mN**2-mphi**2
            a = 1/diff-(mN**2*np.log(mN**2/mphi**2)/(diff)**2)
            amplitude = a
        return amplitude
    C01 = C0(s,mphi1,mN)
    C02 = C0(s,mphi2,mN)
    C01c = np.conjugate(C01)
    C02c = np.conjugate(C02)
    
    PV_terms = (C01+C02)*(C01c+C02c)
    return s*lam**2*Ynu**4*mN**2*PV_terms.real/(512*np.pi**4)

def relic_density_swave(mx,mphi1,del_mphi,mn,lam,Ynu,g_s_new,g_ss_new,g_ss_ln_new,gx):
    mphi2 = mphi1*(1-del_mphi)

    s0 = 4*mx**2
    
    K_0 = 4/(256*np.pi*mx**2)
    
    J_0 = squared_amplitude_s(s0,mphi1,mphi2,mn,lam,Ynu)
    
    a_cm = J_0*K_0
    
    #cross-section coefficients
    sigma_v_rel_in = 2*a_cm      #relative velocity frame = 2*a_cm 
    
    sigmav = float(sigma_v_rel_in)/cgs_to_gev
    
    def x_star_rel_in(x, mx, sigmav, g_ss_new,g_ss_ln_new,gx):
        T = mx/x #in GeVs
        T_ln = np.log(T)
        
        if T<1e-5:
            dof = g_ss_new(1e-5)
            derivative_dof = g_ss_ln_new(np.log(1e-5),1)
        elif T>1.468120e+06:
            dof = g_ss_new(1.468120e+06)
            derivative_dof = g_ss_ln_new(np.log(1.468120e+06),1) 
        else:
            dof = g_ss_new(T)
            derivative_dof = g_ss_ln_new(T_ln,1)
    
        GH = 8e34*mx*sigmav*np.sqrt(x)*np.exp(-x)/np.sqrt(dof)
        #function = x+np.log(x-1.5)-0.5*np.log(x)-20.5 np.log(1e26*sigmav)-np.log(mx)+0.5*np.log(g_ss_new(T))
        function2 = x-3./2.-derivative_dof-GH*(1.+1./3.*derivative_dof)
        return function2
    
    #changed solver from fsoolve to least_squares
    x_s = scipy.optimize.least_squares(x_star_rel_in,10,args=(mx, sigmav, g_ss_new,g_ss_ln_new,gx),bounds=(1,100)).x
    
    sigma_v_cgs = float(sigma_v_rel_in)/cgs_to_gev
    T_s = mx/x_s
    T_ln_s = np.log(T_s)
    
    if T_s<1e-5:
        g_s = g_ss_new(1e-5)
        g_rho = g_s_new(1e-5)
        derivative_dof_s = g_ss_ln_new(np.log(1e-5),1)
    elif T_s>1.468120e+06:
        g_s = g_ss_new(1.468120e+06)
        g_rho = g_s_new(1.468120e+06)
        derivative_dof_s = g_ss_ln_new(np.log(1.468120e+06),1) 
    else:
        g_s = g_ss_new(T_s)
        g_rho = g_s_new(T_s)
        derivative_dof_s = g_ss_ln_new(T_ln_s,1)

    m_pl = 1.220890e+19     #GeV
    rho_c = 1.053672e-5 #h^2 GeV cm^-3
    s = 2891.2 
    delta_s = 0.6180399
   
    #Yeq = (45/(2*np.pi**4))*((np.pi/8)**0.5)*(gx/g_s)*x_s**(3/2)*np.exp(-x_s)
    #Y_s = Yeq*(1+delta_s)
    
    #entropy_s = 2*np.pi**2/45*g_s*T_s**3
    #hubble_s = np.sqrt(8*np.pi**3/90)*g_s/np.sqrt(g_s_new(T_s))*(1+1/3*g_ss_ln_new(T_ln_s,1))/m_pl*T_s**2
    #gamma_s = Y_s*entropy_s*sigma_v_cgs*cgs_to_gev
    #GH_s = gamma_s/hubble_s
    GH_s = (x_s-3/2-derivative_dof_s)/(1+1/3*derivative_dof_s)*(1+1/(delta_s*(2+delta_s)))
    
    def integrand(T,T_s,g_s,g_ss_new,g_ss_ln_new):
        T_ln = np.log(T)
        if T<1e-5:
            g_ss = g_ss_new(1e-5)
            derivative_dof = g_ss_ln_new(np.log(1e-5),1)
        elif T>1.468120e+06:
            g_ss = g_ss_new(1.468120e+06)
            derivative_dof = g_ss_ln_new(np.log(1.468120e+06),1) 
        else:
            g_ss = g_ss_new(T)
            derivative_dof = g_ss_ln_new(T_ln,1)
        
        return 1/T_s*np.sqrt(g_ss/g_s)*(1+1./3.*derivative_dof)
    
    integral = scipy.integrate.quad(integrand,2.3e-13,T_s,args=(T_s,g_s,g_ss_new,g_ss_ln_new))#,epsabs=1e-10,epsrel=1e-9,limit=100)
    a_s = integral[0]
    error = integral[1]
    
    ##Yf = Y_s/(1+a_s*GH_s)
    #omega = mx*Yf*s/rho_c
    #print('parameters: ', mx, mphi1, del_mphi, mn, lam, Ynu)
    #print('values: ',x_s,T_s,sigma_v_cgs, g_s,a_s,GH_s,1+a_s*GH_s)
    
    if sigma_v_cgs==0:
        print("cross-section is zero for values: ",mx,mphi1,del_mphi,mn,lam,Ynu)
    
    omega_h2 = 9.92e-28/(sigma_v_cgs)*x_s/np.sqrt(g_s)*GH_s/(1+a_s*GH_s)
    #print('omega: ', omega_h2)
    return omega_h2, sigma_v_cgs #, x_s, np.sqrt(g_s), GH_s, 50*a_s


#elastic scattering calculations -----------------------------------------------------

#scattering amplitude
def squared_amplitude_sc(t,mx,mphi1,mphi2,mn,lam,Ynu):
    #expansion around t=0 up to t^2 to mimic behaviour at low neutrino energies
    def C0(mphi,mn):
        diff = mn**2-mphi**2
        a = 1/diff-(mn**2*np.log(mn**2/mphi**2)/(diff)**2)
        b = -(2*mn**4+5*mn**2*mphi**2-mphi**4)/(12*mphi**2*(diff)**3)+mn**4*np.log(mn**2/mphi**2)/(2*(diff**4))
        c = -(3*mn**8-27*mn**6*mphi**2-47*mn**4*mphi**4+13*mn**2*mphi**6-2*mphi**8)/(180*mphi**4*diff**5)-mn**6*np.log(mn**2/mphi**2)/(3*diff**6)
        amplitude = a+b*t+c*t**2
        return amplitude
    C01 = C0(mphi1,mn)
    C02 = C0(mphi2,mn)
    C01c = np.conjugate(C01)
    C02c = np.conjugate(C02)
    
    PV_terms = (C01+C02)*(C01c+C02c)
    amp_sq = -0.5*lam**2*Ynu**4*mn**2*t*PV_terms/(512*np.pi**4)
    return amp_sq 
    
#expressions in the lab frame

#differential scattering cross-section
def dsigdcos(x, Enu, mx, mphi1, mphi2, mN, lam, Ynu):
    Enu_p = 1/(1/Enu+1/mx*(1-x)) #outgoing neutrino energy
    s = mx**2 + 2*mx*Enu
    t = -2*Enu*Enu_p*(1-x)
    u = -2*mx*Enu_p+mx**2
    return Enu_p**2/(2*Enu*2*mx*8*np.pi*Enu*mx)*squared_amplitude_sc(t,mx,mphi1,mphi2,mN,lam,Ynu)

#cross-section integrated from -0.95 to 0.95 to avoid collinear singularities   
def sigma_lab(Enu, mx, mphi1, mphi2, mN, lam, Ynu):
    return scipy.integrate.quad(dsigdcos,-0.95,0.95,args=(Enu,mx,mphi1,mphi2,mN,lam,Ynu))[0]

#thermal average with DM at rest and massless neutrinos with FD distribution, DM velocity contribution cancels
def sigmav_sc(Tnu, mx, mphi1, del_mphi, mN, lam, Ynu):
    mphi2 = mphi1*(1-del_mphi)
    def f_nu(x):
        return x**2/(np.exp(x)+1)    
    def sigma_fnu(x,Tnu,mx,mphi1,mphi2,mN,lam,Ynu):
        Enu = x*Tnu
        return f_nu(x)*sigma_lab(Enu,mx,mphi1,mphi2,mN,lam,Ynu)
    numerator = 4*np.pi*Tnu**3*scipy.integrate.quad(sigma_fnu,0,np.inf,args=(Tnu,mx,mphi1,mphi2,mN,lam,Ynu))[0]
    denominator = 4*np.pi*Tnu**3*scipy.integrate.quad(f_nu,0,np.inf)[0] #6*np.pi*scipy.special.zeta(3)*Tnu**3 
    return numerator/denominator

#neutrino mass generation
def mnu(mphi1,del_mphi,mN,Ynu):
    mphi2 = mphi1*(1-del_mphi)
    return Ynu**2/(32*np.pi**2)*mN*(mphi1**2/(mN**2 - mphi1**2)*np.log(mN**2/mphi1**2) - mphi2**2/(mN**2 - mphi2**2)*np.log(mN**2/mphi2**2))

#W decay width to N and phi1
def w_decay_width(mphi,mn,ynu):
    def matfunc23(m12sq,mphi,mn,ynu):
        return -(1/(3*mw**2*sinw**2))*al*np.pi*ynu**2*(-((4*m12sq*(mn**2-mphi**2)*mw**4)/(-m12sq**2-mn**2*(mphi**2-mw**2)+m12sq*(mn**2+mphi**2+mw**2-np.sqrt(m12sq-2*mn**2+mn**4/m12sq)*np.sqrt(m12sq+(mphi**2-mw**2)**2/m12sq-2*(mphi**2+mw**2)))))+(1/(2*m12sq))*(m12sq-mn**2-2*mw**2)*(-m12sq**2-mn**2*(mphi**2-mw**2)+m12sq*(mn**2+mphi**2+mw**2-np.sqrt(m12sq-2*mn**2+mn**4/m12sq)*np.sqrt(m12sq+(mphi**2-mw**2)**2/m12sq-2*(mphi**2+mw**2))))+(-2*m12sq*mw**2+2*mw**2*(mphi**2+mw**2))*np.log((-m12sq**2-mn**2*(mphi**2-mw**2)+m12sq*(mn**2+mphi**2+mw**2-np.sqrt(m12sq-2*mn**2+mn**4/m12sq)*np.sqrt(m12sq+(mphi**2-mw**2)**2/m12sq-2*(mphi**2+mw**2))))/(2*m12sq)))+(1/(3*mw**2*sinw**2))*al*np.pi*ynu**2*((4*m12sq*(mn**2-mphi**2)*mw**4)/(m12sq**2+mn**2*(mphi**2-mw**2)-m12sq*(mn**2+mphi**2+mw**2+np.sqrt(m12sq-2*mn**2+mn**4/m12sq)*np.sqrt(m12sq+(mphi**2-mw**2)**2/m12sq-2*(mphi**2+mw**2))))-(1/(2*m12sq))*(m12sq-mn**2-2*mw**2)*(m12sq**2+mn**2*(mphi**2-mw**2)-m12sq*(mn**2+mphi**2+mw**2+np.sqrt(m12sq-2*mn**2+mn**4/m12sq)*np.sqrt(m12sq+(mphi**2-mw**2)**2/m12sq-2*(mphi**2+mw**2))))+(-2*m12sq*mw**2+2*mw**2*(mphi**2+mw**2))*np.log((-m12sq**2-mn**2*(mphi**2-mw**2)+m12sq*(mn**2+mphi**2+mw**2+np.sqrt(m12sq-2*mn**2+mn**4/m12sq)*np.sqrt(m12sq+(mphi**2-mw**2)**2/m12sq-2*(mphi**2+mw**2))))/(2*m12sq)))
    integral = scipy.integrate.quad(matfunc23,mn**2,(mw-mphi)**2,args=(mphi,mn,ynu))[0]
    return 1/(2*np.pi)**3*1/(32*mw**3)*integral

#Z decay width to N and phi1
def arccoth(x):
    if abs(x) < 1:
        return np.nan  # or any other value to represent undefined
    else:
        return 0.5 * np.log((x + 1) / (x - 1))

def Integral23ratio(m12sq, alpha, beta, ynu):
    return (
        1 / (3 * cosw**2 * mz**2 * sinw**2) *
        al * np.pi * (cosw**2 + sinw**2)**2 * ynu**2 * (
            mz * np.sqrt((m12sq / mz**2 - beta**2)**2 / (m12sq / mz**2)) *
            np.sqrt(m12sq + (mz**2 * (alpha**2 - 1)**2) / (m12sq / mz**2) - 2 * mz**2 * (alpha**2 + 1)) *
            (
                m12sq**2 * mz**2 * (beta**2 - alpha**2) +
                beta**2 * mz**6 * (beta**2 - alpha**2 + 1) * (beta**2 + 2) +
                m12sq / mz**2 * mz**6 * (-2 * beta**4 + 4 * alpha**2 + beta**2 * (2 * alpha**2 - 5))
            ) / (
                m12sq / mz**2 * mz**4 * (beta**2 - alpha**2) -
                beta**2 * mz**4 * (beta**2 - alpha**2 + 1)
            ) +
            4 * mz**4 * (-m12sq / mz**2 + alpha**2 + 1) *
            arccoth(
                (mz**2 * (-m12sq / mz**2 + beta**2 + alpha**2 + 1) +
                 mz**4 / m12sq * beta**2 * (1 - alpha**2)) /
                (
                    mz * np.sqrt((m12sq / mz**2 - beta**2)**2 / (m12sq / mz**2)) *
                     np.sqrt(m12sq + (mz**2 * (alpha**2 - 1)**2) / (m12sq / mz**2) - 2 * mz**2 * (alpha**2 + 1))
                )
            )
        )
    )

def z_decay_width(mphi, mn, ynu):
    mz_mmu_diff = 91.0819416245 #GeV
    mz_mtau_diff = 89.41074     #GeV
    
    def integrand(m12sq):
        alpha = mphi/mz
        beta = mn/mz
        return Integral23ratio(m12sq, alpha, beta, ynu)

    prefactor = 1 / ((2 * np.pi)**3) / (32 * mz**3)
    
    if mphi+mn<mz_mmu_diff and mphi+mn<mz_mtau_diff:
        result, _ = quad(integrand, mn**2, (mz-mphi)**2, epsabs=1.49e-08, epsrel=1.49e-08)
        total_decay = 3*result
    elif mphi+mn<mz_mmu_diff:
        result, _ = quad(integrand, mn**2, (mz-mphi)**2, epsabs=1.49e-08, epsrel=1.49e-08)
        total_decay = 2*result
    elif mphi+mn<mz:
        result, _ = quad(integrand, mn**2, (mz-mphi)**2, epsabs=1.49e-08, epsrel=1.49e-08)
        total_decay = result
    else:
        total_decay = 1e-50 #unconstrained decay
        
    return prefactor * total_decay

#charged meson decay width to N and phi assuming phi massless (conservative bound)

def m_decay_width(m_M,f_M,mn,ynu):
    #G_F g fermi
    #f_M form factor
    prefactor = ynu**2*G_F**2*f_M**2/(768*np.pi**3*m_M**3)
    term1 = (m_M**2-mn**2)*(m_M**4+10*m_M**2*mn**2+mn**4)
    term2 = 12*m_M**2*mn**2*(m_M**2+mn**2)*np.log(m_N/mn)
    return prefactor*(term1-term2)

