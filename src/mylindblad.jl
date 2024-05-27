using DifferentialEquations
using SparseArrays
using LinearAlgebra
using KrylovKit
using Plots

export Lindblad_equations

# 定义Lindblad方程的右端项
function lindblad_rhs!(dρ, ρ, p, t)
    H = p[1]
    C = p[2]
    thisN = size(H)[1]
    
    # println("yes", " ", thisN, " ", size(ρ), " ", size(dρ), " ", t)

    myrho = reshape(ρ, thisN, thisN)
    mydrho = -im * (H * myrho - myrho * H)
    # dρ .= -im * (H * ρ - ρ * H)
    for c in C
        mydrho .+= c * myrho * c' - 0.5 * (c' * c * myrho + myrho * c' * c)
        # dρ .+= c * ρ * c' - 0.5 * (c' * c * ρ + ρ * c' * c)
    end
    dρ .= reshape(mydrho, thisN * thisN)
end

# 将解的向量形式恢复成密度矩阵形式
function reshape_solution(sol, N)
    return [reshape(sol[i], N, N) for i in 1:length(sol)]
end


function Lindblad_equations(rho, H, C, tspan)
    N = size(H, 1)
    # 将密度矩阵平铺成向量形式
    rho0_vec = vec(Complex.(rho))

    # 定义参数列表
    p = [H, C]

    # 定义ODE问题
    prob = ODEProblem(lindblad_rhs!, rho0_vec, tspan, p)

    # 求解ODE问题
    sol = solve(prob)

    rho_t = reshape_solution(sol, N)

    return rho_t[end]
end

function Lindblad_equations_Test(rho, H, C, tspan, tpoints)
    N = size(H, 1)
    # 将密度矩阵平铺成向量形式
    rho0_vec = vec(Complex.(rho))

    # 定义参数列表
    p = [H, C]

    # 定义ODE问题
    prob = ODEProblem(lindblad_rhs!, rho0_vec, tspan, p)

    # 求解ODE问题
    sol = solve(prob, Tsit5(), saveat=tpoints)

    rho_t = reshape_solution(sol, N)

    return rho_t
end



# LindbladTest()


function decompose_density_matrix(rho)
    dense = Matrix(rho)
    eigenvals, eigenvecs = LinearAlgebra.eigen(dense)
    eigenvals = real(eigenvals)
    return eigenvals', eigenvecs
end

# Assume H is independent of Time
function rk4_step(psi, Heff, t, dt)
    k1 = -1im * Heff * psi
    k2 = -1im * Heff * (psi + 0.5 * dt * k1)
    k3 = -1im * Heff * (psi + 0.5 * dt * k2)
    k4 = -1im * Heff * (psi + dt * k3)
    return psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
end

function time_evolution(psi, H, tspan)
    t = tspan[1]
    dt = 0.005
    while(t+dt < tspan[2])
        psi = rk4_step(psi, H, t, dt)
        t += dt
    end
    psi = rk4_step(psi, H, t, tspan[2] - t)
    return psi
end

function QueryNextQuantumJump(psi_init, Heff, dt, r)
    psi_t = psi_init
    psi_ret = psi_init
    t_ret = 0.0
    while(true)
        psi_next_attempt = rk4_step(psi_t, Heff, 0.0, dt)
        psi_next_norm = real(((psi_next_attempt') * (psi_next_attempt))[1,1])

        if t_ret > 8.0 # in case of decoherence free terms.
            psi_ret = psi_next_attempt
            break
        end
        if psi_next_norm > r
            psi_t = psi_ret = psi_next_attempt
            t_ret += dt

        elseif psi_next_norm == r
            psi_ret = psi_next_attempt
            t_ret += dt
            break

        else
            lside = 0
            rside = dt
            while(true)
                t_step = 0.5 * (lside + rside)
                psi_next_attempt = rk4_step(psi_t, Heff, 0.0, t_step)
                psi_next_norm = real((psi_next_attempt' * (psi_next_attempt))[1,1])

                if psi_next_norm > r
                    lside = t_step
                else
                    rside = t_step
                end
                if ( abs(psi_next_norm - r) <= 1e-8 || abs(lside - rside) <= 1e-6 )
                    psi_ret = psi_next_attempt
                    t_ret += t_step
                    break
                end
            end
            break

        end
    end

    return psi_ret, t_ret
end

# write a program to randomly choose an eigenvector according to the probability distribution eigenvals_norm
function RandomlyChoose(Distribution)
    r = rand()
    for i in 1:length(Distribution)
        if r <= sum(Distribution[1:i])
            return i
        end
    end
end

function Stochastic_Master_equation(rho, H, C, tspan)
    eigenvals, eigenvecs = decompose_density_matrix(rho)
    eigenvals_norm = eigenvals / sum(eigenvals)
    
    ntraj = 1500
    cnt = 0
    # psi_traj = zeros(ComplexF64, size(H,1), ntraj)
    rho_ret = zeros(ComplexF64, size(H,1), size(H,1))
    while( (cnt+=1) <= ntraj )
        psi = eigenvecs[:,RandomlyChoose(eigenvals_norm)]
        t = 0.0
        Heff = H - 0.5im * sum([c' * c for c in C])
        while(true)
            dt = 0.005
            r = rand()
            # println("r=", r)
            psi_next, t_step = QueryNextQuantumJump(psi, Heff, dt, r)
            # println("r=", r, " fixed t_step=", t_step)
            if t + t_step < tspan[2]
                t += t_step
                
                prob_C = [real(psi_next' * c' * c * psi_next) for c in C]
                prob_C = prob_C / sum(prob_C)
                c_index = RandomlyChoose(prob_C)
                
                psi = C[c_index] * psi_next
                psi = psi ./ sqrt(real((psi' * psi)[1,1]))
            elseif t + t_step == tspan[2]
                psi = psi_next ./ sqrt(real((psi_next' * psi_next)[1,1]))
                break
            else
                psi = time_evolution(psi, Heff, [t, tspan[2]])
                psi = psi ./ sqrt(real((psi' * psi)[1,1]))
                break
            end
        end
        # psi_traj[:, cnt] = psi
        if(cnt % 100 == 0)
            println("Finish ", cnt, " trajectories\n")
        end
        rho_ret += (psi * psi') / ntraj
    end
    return rho_ret
end


function LindbladTest()
    N = 2
    sigx = Complex.([0.0 1.0; 1.0 0.0])
    sigplus = Complex.([0.0 1.0; 0.0 0.0])
    sigminus = sigplus'

    Omega = 1.0 # Hamiltonian parameter
    Delta = 0.0 # Detuning
    H = Complex.(- Omega / 2.0 .* sigx - Delta .* sigplus * sigminus)

    Gamma = Omega / 6.0 # Dissipation rate

    rho = [0.5 0.0; 0.0 0.5] # initial state

    tpoints = 0:0.005:8.0

    rho_t_lindblad = Lindblad_equations(rho, H, [sqrt(Gamma) * sigminus], (0.0, 8.0))
    rho_t_traj = Stochastic_Master_equation(rho, H, [sqrt(Gamma) * sigminus], (0.0, 8.0))

    print(rho_t_lindblad, "\n", rho_t_traj)

    # p_excited = [real(rho[1, 1]) for rho in rho_t]

    # plot(tpoints, p_excited, xlabel = "Time", ylabel = "Excited state population", title = "Lindblad_equations")

    
end

# LindbladTest()