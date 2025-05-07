# -*- coding: utf-8 -*-
"""
学生代码模板：计算伽马函数 Gamma(a)
"""

import numpy as np
import matplotlib.pyplot as plt
# TODO: 导入数值积分函数 (例如: from scipy.integrate import quad)
# from scipy.integrate import quad
# TODO: 导入可能需要的数学函数 (例如: from math import ...)
# from math import factorial, sqrt, pi, exp, log
from scipy.integrate import quad
from math import factorial, sqrt, pi, exp, log
# --- Task 1: 绘制被积函数 ---

def integrand_gamma(x, a):
    """
    计算伽马函数的原始被积函数: f(x, a) = x^(a-1) * exp(-x)

    Args:
        x (float or np.array): 自变量值。
        a (float): 伽马函数的参数。

    Returns:
        float or np.array: 被积函数在 x 处的值。

    Hints:
        - 需要处理 x=0 的情况 (根据 a 的值可能为 0, 1, 或 inf)。
        - 对于 x > 0, 考虑使用 exp((a-1)*log(x) - x) 来提高数值稳定性。
    """
    # TODO: 实现被积函数的计算逻辑
    if x < 0:
        return 0.0 # 或者抛出错误，因为积分区间是 [0, inf)

    if x == 0:
        # TODO: 处理 x=0 的情况 (考虑 a>1, a=1, a<1)
        # Placeholder
        if a > 1:
            return 0.0
        elif a == 1:
            return 1.0
        else:  # a < 1
            return np.inf
    elif x > 0:
        # TODO: 计算 x > 0 的情况，建议使用 log/exp 技巧
        try:
            # log_f = ...
            # return exp(log_f)
            # Placeholder
            log_f = (a-1)*log(x) - x
            return exp(log_f)
        except ValueError:
            return np.nan
        except ValueError:
            return np.nan # 处理可能的计算错误
    else: # 理论上不会进入这里
        return np.nan

    # 临时返回值，需要替换
    return 0.0


def plot_integrands():
    """绘制 a=2, 3, 4 时的被积函数图像"""
    x_vals = np.linspace(0.01, 10, 400) # 从略大于0开始
    plt.figure(figsize=(10, 6))

    print("绘制被积函数图像...")
    for a_val in [2, 3, 4]:
        print(f"  计算 a = {a_val}...")
        # TODO: 计算 y_vals = [integrand_gamma(x, a_val) for x in x_vals]
        y_vals = np.zeros_like(x_vals) # Placeholder

        # TODO: 绘制曲线 plt.plot(...)
        # plt.plot(x_vals, y_vals, label=f'$a = {a_val}$')
        plt.plot(x_vals, y_vals, label=f'$a = {a_val}$')
        # TODO: (可选) 标记理论峰值位置 x = a-1
        # peak_x = a_val - 1
        # if peak_x > 0:
        #    peak_y = integrand_gamma(peak_x, a_val)
        #    plt.plot(peak_x, peak_y, 'o', ms=5)
        peak_x = a_val - 1
        if peak_x > 0:
            peak_y = integrand_gamma(peak_x, a_val)
            plt.plot(peak_x, peak_y, 'o', ms=5)
   
    plt.xlabel("$x$")
    plt.ylabel("$f(x, a) = x^{a-1} e^{-x}$")
    plt.title("Integrand of the Gamma Function")
    plt.legend() # 需要 plt.plot 中有 label 才会显示
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    # plt.show() # 在 main 函数末尾统一调用 plt.show()

# --- Task 2 & 3: 解析推导 (在注释或报告中完成) ---
# Task 2: 峰值位置推导
# (在此处或报告中写下你的推导过程)
# 结果: x = a - 1

# Task 3: 变量代换 z = x/(c+x)
# 1. 当 z=1/2 时, x = ? (用 c 表示)
#    (在此处或报告中写下你的推导)
#    结果: x = c
# 2. 为使峰值 x=a-1 映射到 z=1/2, c 应取何值? (用 a 表示)
#    (在此处或报告中写下你的推导)
#    结果: c = a - 1

# --- Task 4: 实现伽马函数计算 ---

def transformed_integrand_gamma(z, a):
    """
    计算变换后的被积函数 g(z, a) = f(x(z), a) * dx/dz
    其中 x = cz / (1-z), dx/dz = c / (1-z)^2, 且 c = a-1

    Args:
        z (float or np.array): 变换后的自变量 (积分区间 [0, 1])。
        a (float): 伽马函数的参数。

    Returns:
        float or np.array: 变换后的被积函数值。

    Hints:
        - 这个变换主要对 a > 1 有意义 (c > 0)。需要考虑如何处理 a <= 1 的情况。
        - 计算 f(x(z), a) 时可以调用上面实现的 integrand_gamma 函数。
        - 处理 z=0 和 z=1 的边界情况。
    """
    # TODO: 实现变换后的被积函数计算逻辑

    # 首先处理 c = a-1
    c = a - 1.0
    if c <= 0:
        # 如何处理 a <= 1? 直接返回 0 可能导致积分结果错误。
        # 也许在 gamma_function 中根据 a 的值选择不同的积分方法更好？
        # 暂时返回 0，但需要注意这可能不适用于所有情况。
        # 或者，如果 gamma_function 保证只在 a>1 时调用此函数，则这里可以假设 c>0。
        # return 0.0 # 临时处理
        # 假设调用者保证 a > 1
        if a <= 1: # 增加一个检查
             print(f"警告: transformed_integrand_gamma 假定 a > 1，但接收到 a={a}")
             return np.nan # 或者抛出错误

    # 处理 z 的边界
    if z < 0 or z > 1: return 0.0
    if z == 1: return 0.0 # 对应 x=inf

    # TODO: 计算 x = c*z / (1-z)
    # TODO: 计算 dxdz = c / (1-z)**2
    x = 0 # Placeholder
    dxdz = 0 # Placeholder
    x = c*z / (1-z)
    dxdz = c / (1-z)**2
    # TODO: 计算 f(x, a) * dx/dz，调用 integrand_gamma(x, a)
    val_f = 0 # Placeholder
    result = val_f * dxdz
    if x == 0:
        return 0.0
    
    try:
        log_f = (a-1)*log(x) - x
        val_f = exp(log_f)
    except ValueError:
        return 0.0
    # 检查结果是否有效
    if not np.isfinite(result):
        return 0.0 # 或 np.nan

    return result

def gamma_function(a):
    """
    计算 Gamma(a) 使用数值积分。

    Args:
        a (float): 伽马函数的参数。

    Returns:
        float: Gamma(a) 的计算值。

    Hints:
        - 检查 a <= 0 的情况。
        - 考虑对 a > 1 使用变换后的积分 (transformed_integrand_gamma, 区间 [0, 1])。
        - 考虑对 a <= 1 使用原始积分 (integrand_gamma, 区间 [0, inf])，因为变换推导不适用。
        - 使用导入的数值积分函数 (例如 `quad`)。
    """
    if a <= 0:
        print(f"错误: Gamma(a) 对 a={a} <= 0 无定义。")
        return np.nan

    integral_value = np.nan # Placeholder

    try:
        if a > 1.0:
            # TODO: 使用数值积分计算变换后的积分从 0 到 1
            # integral_value, error = quad(transformed_integrand_gamma, 0, 1, args=(a,))
            # Placeholder
            integral_value, error = quad(transformed_integrand_gamma, 0, 1, args=(a,))
        else: # a <= 1
            # TODO: 使用数值积分计算原始积分从 0 到 inf
            # integral_value, error = quad(integrand_gamma, 0, np.inf, args=(a,))
            # Placeholder
            integral_value, error = quad(integrand_gamma, 0, np.inf, args=(a,))

        # print(f"Integration error estimate for a={a}: {error}") # Optional: print error
        return integral_value

    except Exception as e:
        print(f"计算 Gamma({a}) 时发生错误: {e}")
        return np.nan

# --- 主程序 ---
if __name__ == "__main__":
    # --- Task 1 ---
    print("--- Task 1: 绘制被积函数 ---")
    # plot_integrands() # 取消注释以执行绘图

    # --- Task 2 & 3 ---
    print("\n--- Task 2 & 3: 解析推导见代码注释/报告 ---")
    # (确保注释或报告中有推导)

    # --- Task 4 ---
    print("\n--- Task 4: 测试 Gamma(1.5) ---")
    a_test = 1.5
    # TODO: 调用 gamma_function 计算 gamma_calc
    gamma_calc = 0.0 # Placeholder
    # TODO: 计算精确值 gamma_exact = 0.5 * sqrt(pi)
    gamma_exact = 0.0 # Placeholder
    print(f"计算值 Gamma({a_test}) = {gamma_calc:.8f}")
    print(f"精确值 sqrt(pi)/2 = {gamma_exact:.8f}")
    # TODO: 计算并打印相对误差
    # if gamma_exact != 0:
    #     relative_error = abs(gamma_calc - gamma_exact) / abs(gamma_exact)
    #     print(f"相对误差 = {relative_error:.4e}")

    # --- Task 5 ---
    print("\n--- Task 5: 测试整数 Gamma(a) = (a-1)! ---")
    for a_int in [3, 6, 10]:
        print(f"\n计算 Gamma({a_int}):")
        # TODO: 调用 gamma_function 计算 gamma_int_calc
        gamma_int_calc = 0.0 # Placeholder
        # TODO: 计算精确值 exact_factorial = float(factorial(a_int - 1))
        exact_factorial = 0.0 # Placeholder
        print(f"  计算值 = {gamma_int_calc:.8f}")
        print(f"  精确值 ({a_int-1}!) = {exact_factorial:.8f}")
        # TODO: 计算并打印相对误差
        # if exact_factorial != 0:
        #     relative_error_int = abs(gamma_int_calc - exact_factorial) / abs(exact_factorial)
        #     print(f"  相对误差 = {relative_error_int:.4e}")

    # --- 显示图像 ---
    # plt.show() # 取消注释以显示 Task 1 的图像
