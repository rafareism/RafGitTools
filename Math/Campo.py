#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de campo composto: coerência (phi) × amor (psi) × topologia (g) + memória
Simulação 1D espacial + tempo para o paper "computational field model".
Requer: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# ========== PARÂMETROS (passíveis de varrer – item 16) ==========
N = 20                    # sítios no anel 1D
L = 2*np.pi               # comprimento físico (periódico)
dx = L / N
x = np.linspace(0, L, N, endpoint=False)

omega_phi = 1.0           # frequência natural phi
omega_psi = 1.27          # frequência natural psi (levemente diferente -> batimento)
gamma = 0.05              # amortecimento
lam = 0.1                 # não-linearidade (phi^3, psi^3)
D_phi = 0.2               # intensidade da difusão topológica para phi
D_psi = 0.2               # idem para psi
kappa_cross = 0.3         # força do acoplamento cruzado via memória
tau_mem = 2.0             # escala de tempo da memória
eta_graph = 0.01          # taxa de aprendizado da topologia
decay_graph = 0.005       # decaimento da topologia
g_max = 1.0               # valor máximo da conexão
g_min = 0.0               # valor mínimo
init_amp = 0.3            # amplitude inicial da perturbação local

# Tempo de integração
t_span = (0.0, 200.0)
t_eval = np.linspace(0, 200, 5000)  # amostragem para FFT e gráficos
dt_save = t_eval[1] - t_eval[0]

# ========== INICIALIZAÇÃO ==========
# Campos: phi, dphi, psi, dpsi, mem_phi, mem_psi (cada N)
# Topologia: matriz g (N,N) simétrica, diagonal = 1 (auto-acoplamento não usado na difusão)
g0 = np.zeros((N,N))
for i in range(N):
    g0[i,i] = 1.0          # diagonal não participa da difusão, será mantida
    # vizinhos próximos com conexão moderada inicial
    g0[i, (i+1)%N] = 0.3
    g0[(i+1)%N, i] = 0.3

# Estado inicial: pequenas flutuações aleatórias + pulso local em phi
np.random.seed(42)
phi0 = 0.01 * np.random.randn(N)
dphi0 = np.zeros(N)
psi0 = 0.01 * np.random.randn(N)
dpsi0 = np.zeros(N)
# Perturbação local para excitar o sistema
phi0[N//2] = init_amp
mem_phi0 = np.zeros(N)
mem_psi0 = np.zeros(N)

# Vetor de estado: phi (N), dphi (N), psi (N), dpsi (N), Mphi (N), Mpsi (N), g_flat (N*N)
# Para simplificar, tratamos g como parâmetro evoluindo junto.
g_flat0 = g0.flatten()
state0 = np.concatenate([phi0, dphi0, psi0, dpsi0, mem_phi0, mem_psi0, g_flat0])

# ========== FUNÇÃO DE DERIVADAS ==========
def dynamics(t, state):
    # Desempacotar
    idx = 0
    phi = state[idx:idx+N]; idx += N
    dphi = state[idx:idx+N]; idx += N
    psi = state[idx:idx+N]; idx += N
    dpsi = state[idx:idx+N]; idx += N
    Mphi = state[idx:idx+N]; idx += N
    Mpsi = state[idx:idx+N]; idx += N
    g_flat = state[idx:];  # resto
    g = g_flat.reshape((N,N))

    # Pré-calcular diferenças topológicas (difusão)
    # Para cada sítio i: sum_j g_ij (phi_j - phi_i)
    # = (g @ phi) - (phi * sum_j g_ij)  -> podemos fazer matricial
    g_sum = np.sum(g, axis=1)          # soma das linhas
    diff_phi = g @ phi - phi * g_sum
    diff_psi = g @ psi - psi * g_sum

    # Equações de movimento
    # phi: d²phi/dt² + gamma*dphi/dt + omega_phi^2 * phi + lam*phi^3 + D_phi*diff_phi + kappa*Mpsi = 0
    ddphi = -gamma*dphi - omega_phi**2*phi - lam*phi**3 - D_phi*diff_phi - kappa_cross*Mpsi

    # psi: análogo, acoplado com Mphi
    ddpsi = -gamma*dpsi - omega_psi**2*psi - lam*psi**3 - D_psi*diff_psi - kappa_cross*Mphi

    # Memória: dMphi/dt = (1/tau)*(phi - Mphi)   (kernel exponencial normalizado)
    dMphi = (1.0/tau_mem) * (phi - Mphi)
    dMpsi = (1.0/tau_mem) * (psi - Mpsi)

    # Evolução da topologia (dinâmica lenta, Hebbiana)
    # dg_{ij}/dt = eta*( (phi_i*phi_j + psi_i*psi_j) - decay*g_ij ) , limitada a [0,g_max]
    # Atualizamos apenas elementos fora da diagonal (i != j) para manter simetria
    phi_outer = np.outer(phi, phi)
    psi_outer = np.outer(psi, psi)
    dg = eta_graph * (phi_outer + psi_outer - decay_graph * g)
    # Aplicar limites: saturação suave via taxa; vamos apenas integrar e depois limitar na pós-integração (não aqui para não quebrar a EDO)
    # Mas para estabilidade, podemos usar uma sigmoide; aqui integraremos livre e depois limitamos.
    # Para simetria: dg deve ser simétrico. As contribuições acima são simétricas.
    # Diagonal não deve mudar, mantida =1.
    np.fill_diagonal(dg, 0.0)

    # Montar vetor de derivadas
    deriv = np.concatenate([
        dphi, ddphi,
        dpsi, ddpsi,
        dMphi, dMpsi,
        dg.flatten()
    ])
    return deriv

# ========== INTEGRAÇÃO ==========
print("Integrando... (pode levar ~30s no Termux)")
sol = solve_ivp(dynamics, t_span, state0, t_eval=t_eval, method='RK45',
                rtol=1e-6, atol=1e-8, max_step=0.1)
print("Finalizado.")

# Extrair resultados
phi_t = sol.y[:N, :]        # N x len(t)
psi_t = sol.y[2*N:3*N, :]
Mphi_t = sol.y[4*N:5*N, :]
Mpsi_t = sol.y[5*N:6*N, :]
g_flat_t = sol.y[6*N:, :]
Nt = len(sol.t)

# Reconstruir topologia final (último instante) limitando
g_final = g_flat_t[:,-1].reshape((N,N))
# Aplicar limite [0, g_max] e manter diagonal=1
g_final = np.clip(g_final, g_min, g_max)
np.fill_diagonal(g_final, 1.0)

# ========== GRÁFICOS (itens 11,12,13,15) ==========
# 1) Energia por campo ao longo do tempo (transferência)
E_phi = 0.5 * (omega_phi**2 * phi_t**2 + phi_t**2 + 0.25*lam*phi_t**4)  # simplificado, somado por sítio
E_phi_total = np.sum(0.5 * (omega_phi**2 * phi_t**2 + phi_t**2 + 0.25*lam*phi_t**4), axis=0)
E_psi_total = np.sum(0.5 * (omega_psi**2 * psi_t**2 + psi_t**2 + 0.25*lam*psi_t**4), axis=0)
E_total = E_phi_total + E_psi_total

plt.figure(figsize=(10,4))
plt.plot(sol.t, E_phi_total, label='E φ (coerência)')
plt.plot(sol.t, E_psi_total, label='E ψ (amor)')
plt.plot(sol.t, E_total, 'k--', label='E total')
plt.xlabel('Tempo')
plt.ylabel('Energia')
plt.title('Transferência de energia entre campos (Ressonância)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('transferencia_energia.png')
print("Gráfico 1 salvo: transferencia_energia.png")

# 2) Espectro FFT de um sítio central
site = N//2
dt = t_eval[1] - t_eval[0]
freqs = fftfreq(Nt, dt)[:Nt//2]
fft_phi = np.abs(fft(phi_t[site,:]))[:Nt//2]
fft_psi = np.abs(fft(psi_t[site,:]))[:Nt//2]

plt.figure(figsize=(8,4))
plt.plot(freqs, fft_phi, label='φ sítio %d'%site)
plt.plot(freqs, fft_psi, label='ψ sítio %d'%site)
plt.xlim(0, 3)  # espera-se picos perto de 1.0 e 1.27, e possíveis bandas laterais
plt.xlabel('Frequência')
plt.ylabel('Amplitude FFT')
plt.title('Espectro de frequência (picos não harmônicos?)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('espectro_fft.png')
print("Gráfico 2 salvo: espectro_fft.png")

# 3) Estabilidade numérica: erro relativo da energia total
E_total_ref = E_total[0]
erro_rel = np.abs(E_total - E_total_ref) / (E_total_ref + 1e-12)

plt.figure(figsize=(8,3))
plt.plot(sol.t, erro_rel)
plt.yscale('log')
plt.xlabel('Tempo')
plt.ylabel('Erro relativo da energia total')
plt.title('Estabilidade numérica (divergência de energia)')
plt.grid(True)
plt.tight_layout()
plt.savefig('estabilidade_energia.png')
print("Gráfico 3 salvo: estabilidade_energia.png")

# 4) Topologia final do grafo (matriz g) - visualização
plt.figure(figsize=(6,5))
plt.imshow(g_final, cmap='viridis', vmin=0, vmax=g_max)
plt.colorbar(label='g_ij')
plt.title('Topologia final (g_{ij})')
plt.xlabel('j')
plt.ylabel('i')
plt.tight_layout()
plt.savefig('topologia_final.png')
print("Gráfico 4 salvo: topologia_final.png")

# 5) Comparação com baseline desacoplado (opcional, aqui apenas indicamos)
# Para o paper, você pode rodar com kappa_cross=0, D=0 e g fixo identidade
# Ex.: state0_baseline = ... , integrar sem memória e sem topologia dinâmica,
# plotar lado a lado. Como sugestão, exiba a diferença de espectro.
print("Para baseline, rode novamente com kappa_cross=0 e eta_graph=0 (g fixo).")
