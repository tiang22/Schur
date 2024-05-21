using DifferentialEquations
using SparseArrays
using LinearAlgebra

export Lindblad_equations

# 定义Lindblad方程的右端项
function lindblad_rhs!(dρ, ρ, p, t)
    H = p[1]
    C = p[2]
    thisN = size(H)[1]
    println("yes", " ", thisN, " ", size(ρ), " ", size(dρ), " ", t)
    myrho = reshape(ρ, thisN, thisN)
    mydrho = -im * (H * myrho - myrho * H)
    # dρ .= -im * (H * ρ - ρ * H)
    for c in C
        mydrho .+= c * myrho * c' - 0.5 * (c' * c * myrho + myrho * c' * c)
        # dρ .+= c * ρ * c' - 0.5 * (c' * c * ρ + ρ * c' * c)
    end
    dρ .= vec(mydrho)
end

# 将解的向量形式恢复成密度矩阵形式
function reshape_solution(sol, N)
    return [reshape(sol[i], N, N) for i in 1:length(sol)]
end


function Lindblad_equations(rho, H, C, tspan)
    N = size(H, 1)
    # 将密度矩阵平铺成向量形式
    rho0_vec = vec(rho)

    # 定义参数列表
    p = [H, C]

    # 定义ODE问题
    prob = ODEProblem(lindblad_rhs!, rho0_vec, tspan, p)

    # 求解ODE问题
    sol = solve(prob)

    rho_t = reshape_solution(sol, N)
    print(size(rho_t))

    for i in 1:length(sol)
        println("Time = ", sol.t[i])
        for x in 1:size(rho_t[i])[1]
            for y in 1:size(rho_t[i])[2]
                if rho_t[i, x, y] != 0
                    println("i= ", x, " j= ", y, " value= ", rho_t[i, x, y])
                end
            end
        end
    end

    return rho_t[end]
end


# N = 2

# # 定义稀疏哈密顿量（例如，一个简单的两能级系统）
# ω₀ = 1.0
# H = sparse([1.0 0.0; 0.0 -1.0] * ω₀)

# # 定义稀疏的塌缩算符（例如，自发辐射导致的塌缩）
# γ = 0.1
# C = [sparse([0.0 1.0; 0.0 0.0]) * sqrt(γ)]

# # 定义初始密度矩阵
# ρ₀ = sparse([0.0 0.0; 0.0 1.0])

# rhot = Lindblad_equations(ρ₀, H, C, (0.0, 0.1))


# # 提取结果并进行绘图（例如，计算激发态的布居随时间的变化）
# p_excited = [real(ρ[2, 2]) for ρ in ρt]

# using Plots
# plot(sol.t, p_excited, xlabel="Time", ylabel="Excited state population", title="Lindblad Master Equation Solution")