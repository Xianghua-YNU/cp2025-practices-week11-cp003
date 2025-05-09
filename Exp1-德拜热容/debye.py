import numpy as np
import matplotlib.pyplot as plt

# 物理常数
kB = 1.380649e-23  # 玻尔兹曼常数，单位：J/K

# 样本参数
V = 1000e-6  # 体积，1000立方厘米转换为立方米
rho = 6.022e28  # 原子数密度，单位：m^-3
theta_D = 428  # 德拜温度，单位：K

def integrand(x):
    """被积函数：x^4 * e^x / (e^x - 1)^2"""
    with np.errstate(over='ignore'):
        ex = np.exp(x)
    return (x**4 * ex) / (ex - 1)**2

def gauss_quadrature(f, a, b, n):
    """实现高斯-勒让德积分"""
    x, w = np.polynomial.legendre.leggauss(n)  # 获取 n 个点的节点和权重（区间为 [-1, 1]）
    # 映射到 [a, b]
    t = 0.5 * (b - a) * x + 0.5 * (b + a)
    return 0.5 * (b - a) * np.sum(w * f(t))

def cv(T):
    """计算给定温度T下的热容"""
    if T == 0:
        return 0.0
    x_max = theta_D / T
    integral = gauss_quadrature(integrand, 0, x_max, 100)
    C = 9 * V * rho * kB * (T / theta_D)**3 * integral
    return C

def plot_cv():
    """绘制热容随温度的变化曲线"""
    T_values = np.linspace(1, 600, 300)
    Cv_values = [cv(T) for T in T_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(T_values, Cv_values, label="Debye Heat Capacity")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Heat Capacity (J/K)")
    plt.title("Debye Model: Heat Capacity vs Temperature")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_cv():
    """测试热容计算函数"""
    test_temperatures = [5, 100, 300, 500]
    print("\n测试不同温度下的热容值：")
    print("-" * 40)
    print("温度 (K)\t热容 (J/K)")
    print("-" * 40)
    for T in test_temperatures:
        result = cv(T)
        print(f"{T:8.1f}\t{result:10.3e}")

def main():
    test_cv()
    plot_cv()

if __name__ == '__main__':
    main()
