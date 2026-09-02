import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#from bone_estradiol import bonerepair

# Tempo de simulação
dias = 360
#t = np.linspace(0, dias, 361) # 0 a 360 dias. 
t_span = (0, dias)
t_eval = np.linspace(*t_span, 1000)
# Condições iniciais, vetor com as onze variáveis.
yinit = np.array([
    500000,  # D
    4000.0, # Mo
    0.0,    # M1
    0.0,    # M2
    700.0,  # C1
    14.0,    # C2
    2500.0, # Cm
    250.0,    # Cb <- alvo da análise
    0.0,    # Mc
    0.0,    # Mb
    0.060   # E2
])

# Parâmetros do modelo, argumentos para as EDO's
params_base = {
    'ke_1': 3.0, 'ke_2': 3.0, 'aed': 4.71e6, 'd_o': 0.156, 'k_12': 0.075, 'k_21': 0.005,
    'd_1': 0.121, 'd_2': 0.163, 'k_o': 5e-7, 'k_1': 8.3e-6, 'd_c1': 0.2, 'k_2': 3.72e-6,
    'k_3': 7e-7, 'd_c2': 0.25, 'K_lm': 1e6, 'k_lb': 1e6, 'd_b': 0.15, 'p_cs': 3e-6,
    'q_cd1': 3e-6, 'q_cd2': 2e-6, 'p_bs': 5e-8, 'q_bd': 5e-8, 'kmax': 0.015,
    'Mmax': 6e5, 'k_01': 0.55, 'a_01': 0.01, 'k_02': 0.3, 'a_02': 0.005, 'a_12': 0.025,
    'a_22': 0.1, 'kpm': 0.5, 'apm': 3.162, 'apm1': 13.0, 'dm': 1.0, 'amb1': 0.1,
    'kpb': 0.2202, 'apb': 10.0, 'ae2': 0.5, 'E2max': 0.019, 'de2': 0.03
}
values = np.linspace(0.1, 1.1, 10)
prm = []
# Armazenar resultados de Cb para cada parametro
Cb_all = []

def bonerepair(t, y, p):
  # Parameters of the system of ten equations described above:
  #ke_1 = 3.0
  #ke_2 = 3.0
  #a_ed = 4.71*10**6
  #d_o = 0.156
  #k_12 = 0.075
  #k_21 = 0.005
  #d_1 = 0.121
  #d_2 = 0.163
  #k_o = 0.0000005
  #k_1 = 0.0000083
  #d_c1 = 0.2 
  #k_2 = 0.00000372
  #k_3 = 0.0000007
  #d_c2 = 0.25
  #K_lm = 1000000.0
  #k_lb = 1000000.0
  #d_b = 0.15
  #p_cs = 0.000003
  #q_cd1 = 0.000003
  #q_cd2 = 0.000002
  #p_bs = 0.00000005
  #q_bd = 0.00000005
  #aed = 4.71*10**6
  #kmax = 0.015
  #Mmax = 6.0*10**5
  #k_01 = 0.55
  #a_01 = 0.01
  #k_02 = 0.3
  #a_02 = 0.005
  #a_12 = 0.025
  #a_22 = 0.1
  #kpm = 0.5
  #apm = 3.162
  #apm1 = 13.0
  #dm = 1.0
  #amb1 = 0.1
  #kpb = 0.2202
  #apb = 10.0
  #ae2 = 0.5 
  #E2max = 0.019 
  #de2 = 0.03  

  # Decouple to simplify the writing of the equations.
  [D, Mo, M1, M2, C1, C2, Cm, Cb, Mc, Mb, E2] = y

  #[
  #  ke_1, ke_2, aed, d_o, k_12, k_21, d_1, d_2, k_o, k_1, d_c1, k_2, k_3, d_c2,
  #  K_lm, k_lb, d_b, p_cs, q_cd1, q_cd2, p_bs, q_bd, kmax, Mmax, k_01, a_01,
  #  k_02, a_02, a_12, a_22, kpm, apm, apm1, dm, amb1, kpb, apb, ae2, E2max, de2
  # ] = p
  
  # Phagocytosis rate
  def RD(t,D,aed):
    return (D/(aed+D))

  # Migration rate of M_o to the injury site
  def RM(t,Mo,M1,M2,kmax,Mmax):
    M = Mo+M1+M2
    return kmax*(1-M/Mmax)

  # Rate of differentiation of M_o into M_1
  def G1(t,k_01,a_01,C1):
    return k_01 * (C1/(a_01+C1))

  # Rate of differentiation of M_1 into M_0
  def G2(t,k_02,a_02,C2):
    return k_02 * (C2/(a_02+C2))

  # Inhibitory effect of the pro-inflammatory cytokine
  def H1(t,a_12,C2):
    return (a_12/(a_12+C2))

  # Inhibitory effect of the anti-inflammatory cytokine
  def H2(t,a_22,C2):
    return (a_22/(a_22+C2))

  # Inhibitory effect of the anti-inflammatory cytokine
  def Am(t,kpm,apm,apm1,C1):
    return kpm * ((apm*apm) + (apm1*C1)) / ((apm*apm) + (C1*C1))

  # Proliferation of C_m inhibited by c_1
  def F1(t,dm,amb1,C1):
    return dm * (amb1/(amb1+C1))

  # Proliferation of C_B inhibited by c_1
  def Ab(t,kpb,apb,C1):
    return kpb * (apb/(apb+C1))
  
  # define the equations:
  dDdt  = -RD(t,D,p['aed'])*(p['ke_1']*M1 + p['ke_2']*M2)
  dModt = RM(t,Mo,M1,M2,p['kmax'],p['Mmax']) - G1(t,p['k_01'],p['a_01'],C1)*Mo - G2(t,p['k_02'],p['a_02'],C2)*Mo - (p['d_o']*Mo)
  dM1dt  = G1(t,p['k_01'],p['a_01'],C1)*Mo + (p['k_21']*M2) - (p['k_12']*M1) - (p['d_1']*M1)
  dM2dt  = G2(t,p['k_02'],p['a_02'],C2)*Mo + (p['k_12']*M1) - (p['k_21']*M2) - (p['d_2']*M2)
  dC1dt  = H1(t,p['a_12'],C2)*(p['k_o']*D + p['k_1']*M1) - (p['d_c1']*C1)*E2
  dC2dt  = H2(t,p['a_22'],C2)*(p['k_2']*M2 + p['k_3']*Cm)*E2 - (p['d_c2']*C2)
  dCmdt  = Am(t,p['kpm'],p['apm'],p['apm1'],C1)*Cm*(1 - (Cm/p['K_lm'])) - (F1(t,p['dm'],p['amb1'],C1)*Cm)
  dCbdt  = Ab(t,p['kpb'],p['apb'],C1)*Cb*(1 - (Cb/p['k_lb'])) + (F1(t,p['dm'],p['amb1'],C1)*Cm) - (p['d_b']*Cb)
  dMcdt  = (p['p_cs'] - p['q_cd1']*Mc)*Cm - (p['q_cd2']*Mc*Cb)
  dMbdt  = (p['p_bs'] - (p['q_bd']*Mb))*Cb
  dE2dt  = ((p['E2max']-E2)/p['E2max'])*p['ae2']*E2 - p['de2']*E2

  # Recombines to return
  dydt = [dDdt, dModt, dM1dt, dM2dt, dC1dt, dC2dt, dCmdt, dCbdt, dMcdt, dMbdt, dE2dt]
  return dydt

# modelo é simulado 10 vezes, cada uma com um valor diferente de cada parametro, para ver como isso afeta a variável Cb.
i=0

for nome_params in params_base:
    fig, axs = plt.subplots(4, 3, figsize=(12, 10))  # 4x3 para acomodar as 11 variáveis
    axs = axs.flatten()

    for v in values:
        params = params_base.copy()
        params[nome_params] *= v
        sol = solve_ivp(lambda t, y: bonerepair(t, y, params), t_span, yinit, t_eval=t_eval)

        for i in range(11):  # só plota as 11 variáveis
            axs[i].plot(sol.t, sol.y[i], label=f'{nome_params} x{v}')
            axs[i].set_title(['Detrito','M0','M1','M2','C1','C2','Cm','Cb','Mc','Mb','E2'][i])
            axs[i].set_xlabel('Tempo')
            axs[i].set_ylabel('Concentração')

    # Remove o subplot vazio
    axs[11].set_visible(False)

    # Coloca a legenda uma vez só
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=5, fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # espaço para a legenda
    plt.savefig(f'Sensibilidade_{nome_params}.png', dpi=300)
    plt.close(fig)




      

