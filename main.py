#a versao de python utilizada foi a 3.11; instalou-se as bibliotecas numpy, scipy, matplotlib; ao longo do código encontram-se comentarios explicativos do mesmo e o racicionio em causa
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import matplotlib.pyplot as plt

# ================= CONFIGURAÇÕES DO SISTEMA =================
np.random.seed(42)
RHO = 1000.0
G = 9.81
ETA = 0.65
A = 185.0
H_MIN, H_MAX = 2.0, 7.0  # Limites operacionais do reservatório
H0 = 4.0
TIME_HORIZON = 24
Q_P_MAX = 120
Q_P_MIN = 40

# Parâmetros hidráulicos
F = 0.02
D = 0.3
L_PR = 2500
L_RF = 5000

# Tarifário energético (€/kWh)
TARIFAS = np.array([
    0.0713, 0.0713, 0.0651, 0.0651, 0.0593, 0.0593,
    0.0778, 0.0778, 0.0851, 0.0851, 0.0923, 0.0923,
    0.0968, 0.0968, 0.10094, 0.10094, 0.10132, 0.10132,
    0.10226, 0.10226, 0.10189, 0.10189, 0.10132, 0.10132
])

# ================= MODELO HIDRÁULICO =================
def altura_bomba(Q):
    return 260 - 0.002 * Q**2

def perdas_carga(Q, L):
    Q_m3s = Q / 3600
    return (32 * F * L) / (D**5 * G * np.pi**2) * Q_m3s**2

def altura_total(Q):
    return 150 + altura_bomba(Q) + perdas_carga(Q, L_PR) + perdas_carga(Q, L_RF)

def potencia_bomba(Q):
    Q_m3s = Q / 3600
    H = altura_total(Q)
    return (RHO * G * Q_m3s * H) / (ETA * 1000)

# ================= MODELO DE CONSUMO =================
def consumo_R(t):
    return -0.004*t**3 + 0.09*t**2 + 0.1335*t + 20

def consumo_VC_max(t):
    t_arr = np.asarray(t)
    return (-1.19333e-7*t_arr**7 - 4.90754e-5*t_arr**6 + 3.733e-3*t_arr**5 
            - 0.09621*t_arr**4 + 1.03965*t_arr**3 - 3.8645*t_arr**2 - 1.0124*t_arr + 75.393)

def consumo_VC_min(t):
    t_arr = np.asarray(t)
    return (1.19333e-7*t_arr**7 - 6.54846e-5*t_arr**6 + 4.1432e-3*t_arr**5 
            - 0.100585*t_arr**4 + 1.05575*t_arr**3 - 3.85966*t_arr**2 - 1.32657*t_arr + 75.393)

# ================= SIMULAÇÃO COM PENALIZAÇÕES CORRIGIDA =================
def simular_sistema_com_penalizacoes(Q_P, cenario, aceitar_violacoes=False):
    """
    Versão corrigida do cálculo de penalizações:
    - Cada hora de violação adiciona 5€
    - Horas consecutivas não acumulam penalizações adicionais
    """
    t = np.arange(TIME_HORIZON)
    volume = A * H0
    volumes = []
    custo_acumulado = 0
    custos = []
    penalizacoes = 0
    violacoes = 0
    
    for hora in range(TIME_HORIZON):
        if cenario == 'max':
            Q_VC = consumo_VC_max(hora)
        else:
            Q_VC = consumo_VC_min(hora)
        Q_R = consumo_R(hora)
        
        if Q_P[hora] > 0:
            volume += Q_P[hora] - Q_VC - Q_R
            custo_acumulado += potencia_bomba(Q_P[hora]) * TARIFAS[hora]
        else:
            volume -= Q_VC + Q_R
        
        # Verificação de violações - CORREÇÃO: penalização fixa de 5€ por hora de violação
        violacao = False
        if volume < A*H_MIN or volume > A*H_MAX:
            violacoes += 1
            violacao = True
            
            if aceitar_violacoes:
                penalizacoes += 5  # CORREÇÃO: Sempre 5€ por hora de violação
            else:
                # Forçar cumprimento das regras
                volume = np.clip(volume, A*H_MIN, A*H_MAX)
        
        volumes.append(volume)
        custos.append(custo_acumulado)
    
    custo_total = custo_acumulado + penalizacoes
    
    return {
        'volumes': np.array(volumes),
        'Q_P': Q_P,
        'custos': np.array(custos),
        'custo_total': custo_total,
        'penalizacoes': penalizacoes,
        'violacoes': violacoes,
        'aceitar_violacoes': aceitar_violacoes
    }

# ================= OTIMIZAÇÃO =================
def otimizar_conservadora(cenario):
    """Estratégia que sempre cumpre as regras"""
    def objetivo(Q_P):
        sim = simular_sistema_com_penalizacoes(Q_P, cenario, False)
        return sim['custos'][-1]
    
    def restricao_volume(Q_P):
        volumes = simular_sistema_com_penalizacoes(Q_P, cenario, False)['volumes']/A
        return volumes
    
    constraints = NonlinearConstraint(restricao_volume, H_MIN, H_MAX)
    bounds = Bounds([0]*24, [Q_P_MAX]*24)
    
    # Chute inicial: ligar nas 8 horas com tarifa mais baixa
    tarifas_ordenadas = np.argsort(TARIFAS)
    Q_P0 = np.zeros(24)
    Q_P0[tarifas_ordenadas[:8]] = Q_P_MAX
    
    res = minimize(
        objetivo,
        Q_P0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'disp': False}
    )
    return res.x

def otimizar_arriscada(cenario):
    """Estratégia que minimiza custo energético, aceitando penalizações"""
    def objetivo(Q_P):
        sim = simular_sistema_com_penalizacoes(Q_P, cenario, True)
        # Minimiza apenas o custo energético (as penalizações são consequência)
        return sim['custos'][-1]
    
    bounds = Bounds([0]*24, [Q_P_MAX]*24)
    
    # Chute inicial: ligar apenas nas 4 horas mais baratas
    tarifas_ordenadas = np.argsort(TARIFAS)
    Q_P0 = np.zeros(24)
    Q_P0[tarifas_ordenadas[:4]] = Q_P_MAX
    
    res = minimize(
        objetivo,
        Q_P0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 500, 'disp': False}
    )
    return res.x

def otimizar_hibrida(cenario):
    """Estratégia que balanceia custos energéticos e penalizações"""
    def objetivo(Q_P):
        sim = simular_sistema_com_penalizacoes(Q_P, cenario, True)
        # Ponderação entre custo energético e penalizações
        return sim['custos'][-1] + 2*sim['penalizacoes']  # Peso menor nas penalizações
    
    def restricao_volume(Q_P):
        volumes = simular_sistema_com_penalizacoes(Q_P, cenario, True)['volumes']/A
        return np.clip(volumes, H_MIN*0.8, H_MAX*1.2)  # Permite pequenas violações
    
    constraints = NonlinearConstraint(restricao_volume, H_MIN*0.8, H_MAX*1.2)
    bounds = Bounds([0]*24, [Q_P_MAX]*24)
    
    # Chute inicial: ligar nas 6 horas mais baratas
    tarifas_ordenadas = np.argsort(TARIFAS)
    Q_P0 = np.zeros(24)
    Q_P0[tarifas_ordenadas[:6]] = Q_P_MAX
    
    res = minimize(
        objetivo,
        Q_P0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'disp': False}
    )
    return res.x

# ================= VISUALIZAÇÃO =================
def plot_resultados(Q_P, cenario, resultado):
    t = np.arange(TIME_HORIZON)
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 14))
    
    # Gráfico 1: Vazão e nível
    ax1 = axs[0]
    ax1.bar(t, resultado['Q_P'], color='r', alpha=0.6, label='Vazão da bomba')
    ax1.set_ylabel('Vazão (m³/h)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.legend(loc='upper left')
    
    ax1b = ax1.twinx()
    ax1b.plot(t, resultado['volumes']/A, 'b-', label='Nível do reservatório')
    ax1b.axhline(H_MIN, color='k', linestyle='--', label='Limites')
    ax1b.axhline(H_MAX, color='k', linestyle='--')
    ax1b.set_ylabel('Nível (m)', color='b')
    ax1b.tick_params(axis='y', labelcolor='b')
    ax1b.legend(loc='upper right')
    
    # Gráfico 2: Consumos
    ax2 = axs[1]
    if cenario == 'max':
        Q_VC = consumo_VC_max(t)
    else:
        Q_VC = consumo_VC_min(t)
    ax2.plot(t, Q_VC, 'm-', label='Consumo VC')
    ax2.plot(t, consumo_R(t), 'g-', label='Consumo R')
    ax2.set_ylabel('Vazão (m³/h)')
    ax2.legend()
    
    # Gráfico 3: Potência
    ax3 = axs[2]
    potencia = [potencia_bomba(Q) if Q > 0 else 0 for Q in resultado['Q_P']]
    ax3.plot(t, potencia, 'k-', label='Potência da bomba')
    ax3.set_ylabel('Potência (kW)')
    ax3.legend()
    
    # Gráfico 4: Custos
    ax4 = axs[3]
    ax4.plot(t, resultado['custos'], 'k-', label='Custo energético')
    if resultado['penalizacoes'] > 0:
        ax4.plot(t, resultado['custos'] + np.linspace(0, resultado['penalizacoes'], 24), 
                 'r--', label='Custo total (com penalizações)')
    ax4.set_ylabel('Custo (€)')
    ax4.set_xlabel('Hora do dia')
    ax4.legend()
    
    plt.suptitle(f"Cenário {cenario.upper()} - Estratégia {'Flexível' if resultado['aceitar_violacoes'] else 'Conservadora'}\n"
    f"Custo Total: €{resultado['custo_total']:.2f}")
    plt.tight_layout()
    plt.savefig(f'resultados_{cenario}_{"flex" if resultado["aceitar_violacoes"] else "cons"}.png', dpi=300)
    plt.ylim(bottom=0)
    plt.show()

# ================= ANÁLISE COMPARATIVA =================
def analisar_cenario(cenario):
    print(f"\n=== ANÁLISE PARA CENÁRIO {cenario.upper()} ===")
    
    # Conservadora (nunca viola os limites)
    Q_P_cons = otimizar_conservadora(cenario)
    res_cons = simular_sistema_com_penalizacoes(Q_P_cons, cenario, False)
    print("\nESTRATÉGIA CONSERVADORA (SEM VIOLAÇÕES):")
    print(f"Custo energético: €{res_cons['custos'][-1]:.2f}")
    print(f"Custo total: €{res_cons['custo_total']:.2f}")
    plot_resultados(Q_P_cons, cenario, res_cons)
    
    # Arriscada (minimiza custo energético, aceita penalizações)
    Q_P_arriscada = otimizar_arriscada(cenario)
    res_arriscada = simular_sistema_com_penalizacoes(Q_P_arriscada, cenario, True)
    print("\nESTRATÉGIA ARRISCADA (MINIMIZA CUSTO ENERGÉTICO):")
    print(f"Custo energético: €{res_arriscada['custos'][-1]:.2f}")
    print(f"Penalizações: €{res_arriscada['penalizacoes']:.2f} (€5/hora violada)")
    print(f"Custo total: €{res_arriscada['custo_total']:.2f}")
    print(f"Violacoes: {res_arriscada['violacoes']} horas")
    plot_resultados(Q_P_arriscada, cenario, res_arriscada)
    
    # Híbrida (balanceia custos e penalizações)
    Q_P_hibrida = otimizar_hibrida(cenario)
    res_hibrida = simular_sistema_com_penalizacoes(Q_P_hibrida, cenario, True)
    print("\nESTRATÉGIA HÍBRIDA (BALANCEADA):")
    print(f"Custo energético: €{res_hibrida['custos'][-1]:.2f}")
    print(f"Penalizações: €{res_hibrida['penalizacoes']:.2f} (€5/hora violada)")
    print(f"Custo total: €{res_hibrida['custo_total']:.2f}")
    print(f"Violacoes: {res_hibrida['violacoes']} horas")
    plot_resultados(Q_P_hibrida, cenario, res_hibrida)
    
    # Determinar melhor estratégia
    estrategias = {
        'conservadora': res_cons['custo_total'],
        'arriscada': res_arriscada['custo_total'],
        'hibrida': res_hibrida['custo_total']
    }
    melhor = min(estrategias, key=estrategias.get)
    
    print(f"\nCONCLUSÃO PARA CENÁRIO {cenario.upper()}:")
    print(f"Melhor estratégia: {melhor.upper()} (€{estrategias[melhor]:.2f})")
    
    if melhor != 'conservadora':
        economia = res_cons['custo_total'] - estrategias[melhor]
        print(f"Economia de €{economia:.2f} em relação à estratégia conservadora")
        print(f"Mas com {res_hibrida['violacoes'] if melhor == 'hibrida' else res_arriscada['violacoes']} horas de violação")

# ================= EXECUÇÃO PRINCIPAL =================
if __name__ == "__main__":
    print("=== ANÁLISE DE ESTRATÉGIAS DE OPERAÇÃO ===")
    print("=== Tarefa 4.2 - Vale a pena cumprir as regras? ===\n")
    print("PENALIZAÇÕES: €5 por cada hora com violação dos limites (não cumulativo)\n")
    
    # Analisar ambos os cenários
    analisar_cenario('min')
    analisar_cenario('max')