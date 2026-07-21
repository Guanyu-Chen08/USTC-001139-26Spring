import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import datetime
import random

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

current_time = datetime.datetime.now()
SAVE_DIR = f"results_{current_time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"[*] 所有运行结果图保存在文件夹: {SAVE_DIR}")

def save_plot(filename):
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

def basic_sir_ode(t, y, beta, gamma):
    S, I, R = np.maximum(y, 0.0)
    return [-beta*S*I, beta*S*I - gamma*I, gamma*I]

def run_part1_basic_sir():
    print("\n--- 正在运行 任务1: 基本SIR模型 ---")
    N, t_span, t_eval = 100000, (0, 150), np.linspace(0, 150, 500)
    
    params_set = [
        (1.07e-5, 1/14.0, N-10, "A: 高传播率, 长感染期"), 
        (4.00e-6, 1/5.0,  N-10, "B: 低传播率, 短感染期"),
        (2.14e-6, 1/14.0, N-10, "C: 较低传播率, 长感染期"),
        (1.07e-5, 1/14.0, 5000, "D: 初始易感极低 (群体免疫)")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (beta, gamma, S0, title) in enumerate(params_set):
        ax = axes.flatten()[i]
        
        sol = solve_ivp(basic_sir_ode, t_span, [S0, 10, N-S0-10], 
                        args=(beta, gamma), t_eval=t_eval, method='RK45')
        
        ax.plot(sol.t, sol.y[0], label='S(t) 易感者', color='blue', lw=2)
        ax.plot(sol.t, sol.y[1], label='I(t) 感染者', color='red', lw=2)
        ax.plot(sol.t, sol.y[2], label='R(t) 移除者', color='green', lw=2)
        
        ax.set_title(f"{title}\n$\\beta={beta:.2e}, \\gamma={gamma:.3f}, S(0)={S0}$")
        ax.set_xlabel("时间 $t$ (天)"), ax.set_ylabel("人口数量")
        ax.legend(), ax.grid(True, ls='--')
        
    plt.tight_layout()
    save_plot("Task1_Basic_SIR_4Sets.png")

def demographic_sir_ode(t, y, beta, gamma, mu, N):
    S, I, R = np.maximum(y, 0.0)
    return [mu*N - beta*S*I - mu*S, beta*S*I - gamma*I - mu*I, gamma*I - mu*R]

def run_part2_demographic_sir():
    print("\n--- 正在运行 任务2: 加入人口动态的SIR模型 ---")
    N, gamma, t_span, t_eval = 1000000, 365.0/14.0, (0, 80), np.linspace(0, 80, 2000)
    
    params_set = [
        (15.0, 1/70.0, "A: 标准人口更替"),
        (15.0, 1/20.0, "B: 高速人口更替"),
        (5.0,  1/70.0, "C: 较低传染率"),
        (30.0, 1/70.0, "D: 极高传染率")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (R0, mu, title) in enumerate(params_set):
        ax1 = axes.flatten()[i]
        beta = R0 * (gamma + mu) / N
        sol = solve_ivp(demographic_sir_ode, t_span, [N*0.05, 1000, N*0.95-1000], 
                        args=(beta, gamma, mu, N), t_eval=t_eval, method='LSODA')
        
        ax2 = ax1.twinx()
        l1 = ax1.plot(sol.t, sol.y[0], label='S(t)', color='blue', alpha=0.7)
        l2 = ax1.plot(sol.t, sol.y[2], label='R(t)', color='green', alpha=0.7)
        l3 = ax2.plot(sol.t, sol.y[1], label='I(t) [右对数轴]', color='red', lw=1.5)
        
        ax1.set_ylabel("S, R 人数"), ax2.set_ylabel("I 人数 (Log)"), ax2.set_yscale('log')
        ax1.set_title(f"{title}\n$\\beta={beta:.2e}, \\mu={mu:.3f}$")
        ax1.grid(True, ls='--')
        lns = l1 + l2 + l3
        ax1.legend(lns, [l.get_label() for l in lns], loc='upper right')
        
    plt.tight_layout()
    save_plot("Task2_Demographic_SIR_4Sets.png")

def seasonal_sir_ode(t, y, beta0, alpha, gamma, mu, N):
    S, I, R = np.maximum(y, 0.0)
    beta_t = beta0 * (1.0 + alpha * np.cos(2.0 * np.pi * t))
    return [mu*N - beta_t*S*I - mu*S, beta_t*S*I - gamma*I - mu*I, gamma*I - mu*R]

def run_part3_seasonal_sir():
    print("\n--- 正在运行 任务3: 季节性传播率模型 ---")
    N, mu, gamma, R0 = 1000000, 1.0/70.0, 365.0/14.0, 15.0
    beta0 = R0 * (gamma + mu) / N
    y0 = [N*0.05, 100, N*0.95-100]
    alphas = [0.05, 0.15, 0.25]
    
    fig_full, axes_full = plt.subplots(3, 1, figsize=(12, 10))
    fig_steady, axes_steady = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, alpha in enumerate(alphas):
        sol = solve_ivp(seasonal_sir_ode, (0, 100), y0, 
                        args=(beta0, alpha, gamma, mu, N), 
                        t_eval=np.linspace(0, 100, 5000), method='LSODA', rtol=1e-8)
        
        axes_full[i].plot(sol.t, sol.y[1], color='red', lw=1.5)
        axes_full[i].set_title(f"全时段演化: $\\alpha$={alpha}")
        axes_full[i].set_ylabel("感染者 I(t)"), axes_full[i].grid(True, ls='--')
        
        mask = sol.t >= 80
        axes_steady[i].plot(sol.t[mask], sol.y[1][mask], color='purple', lw=2)
        axes_steady[i].set_title(f"后期周期结构 (80-100年): $\\alpha$={alpha}")
        axes_steady[i].set_ylabel("感染者 I(t)"), axes_steady[i].grid(True, ls='--')

    axes_full[2].set_xlabel("时间 $t$ (年)")
    axes_steady[2].set_xlabel("时间 $t$ (年)")
    
    fig_full.tight_layout()
    fig_full.savefig(os.path.join(SAVE_DIR, "Task3_Seasonal_Global_Full.png"), dpi=300)
    plt.close(fig_full)
    
    fig_steady.tight_layout()
    fig_steady.savefig(os.path.join(SAVE_DIR, "Task3_Seasonal_Steady_State.png"), dpi=300)
    plt.close(fig_steady)
    
    points_alpha, points_I = [], []
    for alpha in np.linspace(0, 0.3, 60):
        sol = solve_ivp(seasonal_sir_ode, (0, 100), y0, args=(beta0, alpha, gamma, mu, N), 
                        t_eval=np.arange(80, 100, 1), method='Radau', rtol=1e-6)
        for I_val in sol.y[1]:
            points_alpha.append(alpha), points_I.append(I_val)
            
    plt.figure(figsize=(12, 6))
    plt.scatter(points_alpha, points_I, s=8, color='black', alpha=0.6)
    plt.title("季节强度的分岔图")
    plt.xlabel("季节性强度 $\\alpha$"), plt.ylabel("每年同时间感染人数 $I(t)$")
    plt.grid(True, ls='--')
    save_plot("Task3_Bifurcation_Diagram.png")

def run_gillespie(N, T_max, beta0, gamma, mu, init_S, init_I):
    S, I, R = init_S, init_I, N - init_S - init_I
    t, times, I_history = 0.0, [0.0], [I]
    
    while t < T_max:
        rates = [mu*N, beta0*S*I, gamma*I, mu*S, mu*I, mu*R]
        rate_sum = sum(rates)
        if rate_sum == 0 or I == 0:
            times.append(T_max); I_history.append(0)
            break
        t += -np.log(random.random()) / rate_sum
        
        cumulative_rate, threshold = 0.0, random.random() * rate_sum
        for idx, r in enumerate(rates):
            cumulative_rate += r
            if cumulative_rate >= threshold:
                if idx == 0: S += 1
                elif idx == 1: S -= 1; I += 1
                elif idx == 2: I -= 1; R += 1
                elif idx == 3: S -= 1
                elif idx == 4: I -= 1
                elif idx == 5: R -= 1
                break
        times.append(t); I_history.append(I)
    return np.array(times), np.array(I_history)

def run_part4_gillespie_stochastic():
    print("\n--- 正在运行 任务4: Gillespie随机模型 ---")
    gamma, mu, R0 = 365.0/14.0, 1.0/70.0, 15.0
    configs = [(1000, 5), (5000, 5), (50000, 5), (5000, 1), (5000, 20), (5000, 100)]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, (N, init_I) in enumerate(configs):
        ax = axes.flatten()[idx]
        beta = R0 * (gamma + mu) / N
        init_S = int(N * 0.1)
        
        extinct_count = 0
        extinct_times = []
        
        for _ in range(50):
            t_arr, I_arr = run_gillespie(N, 5, beta, gamma, mu, init_S, init_I)
            ax.plot(t_arr, I_arr, color='gray', alpha=0.3, lw=0.8)
            
            if I_arr[-1] == 0:
                extinct_count += 1
                ext_time = t_arr[np.where(I_arr == 0)[0][0]]
                extinct_times.append(ext_time)
                
        sol_det = solve_ivp(demographic_sir_ode, (0, 5), [init_S, init_I, N - init_S - init_I], 
                            args=(beta, gamma, mu, N), t_eval=np.linspace(0, 5, 1000))
        ax.plot(sol_det.t, sol_det.y[1], color='red', lw=2, label="确定性ODE")
        
        ax.set_title(f"N={N}, 初始$I_0$={init_I}")
        ax.set_xlabel("时间 $t$ (年)"), ax.set_ylabel("感染者 $I(t)$"), ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        
        avg_ext_time = np.mean(extinct_times) if extinct_count > 0 else 0.0
        print(f"组别 N={N:<5}, I0={init_I:<3} | 50次灭绝数: {extinct_count} ({extinct_count/50*100:.1f}%) | 平均灭绝时间: {avg_ext_time:.3f} 年")
        
    plt.tight_layout()
    save_plot("Task4_Gillespie_Stochastic_6Sets.png")

if __name__ == "__main__":
    run_part1_basic_sir()
    run_part2_demographic_sir()
    run_part3_seasonal_sir()
    run_part4_gillespie_stochastic()