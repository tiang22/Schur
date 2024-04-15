struct TimeDependent{FT}
    f::FT
end
function qeye(::Type{T}, n::Int) where T
    return Diagonal(ones(T, n))
end
qeye(n::Int) = qeye(ComplexF64, n)

function destroy(::Type{T}, n::Int) where T
    return sparse(1:n-1, 2:n, T.(sqrt.(1:n-1)), n, n)
end
destroy(n::Int) = destroy(ComplexF64, n)

function basis(::Type{T}, n::Int, k::Int) where T
    return SparseVector(n, [k+1], [one(T)])
end
basis(n::Int, k::Int) = basis(ComplexF64, n, k)
expect(op, rho) = tr(op * rho)

# Define Qubit Supspace Projection Operator
projector(t::AbstractVector{T}) where T = OuterProduct(t, t)

function lindblad_diffeq!(drho, rho, H, c_ops)
    Hrho = H * rho
    drho .= -1im .* (Hrho .- Hrho')
    for c in c_ops
        cch = rmul!(c' * c, -0.5)
        cchrho = cch * rho
        drho .+= c * ((rho + rho') ./ 2) * c' + cchrho + cchrho'
    end
    return drho
end

# function tocomplexmatrix(v::AbstractVector{T}) where T<:Real
#     cv = reinterpret(Complex{T}, v)
#     n = isqrt(length(cv))
#     return reshape(cv, n, n)
# end
# function torealvector(m::AbstractMatrix{Complex{T}}) where T
#     return reinterpret(T, vec(m))
# end

struct MesolveRes{MT<:AbstractMatrix}
    times::Vector{Float64}
    e_ops_values::Vector{Vector{ComplexF64}}
    rho::MT
end

function mesolve(H, rho0::AbstractMatrix{<:Complex}, tspan; c_ops, e_ops, solver=:dp5, integrator_config=(;))
    times = Float64[]
    expect_values = Vector{ComplexF64}[]
    function f!(t, u::AbstractMatrix{<:ComplexF64}, du::AbstractMatrix{<:ComplexF64})
        n = isqrt(length(u))
        mdu, mu = reshape(du, n, n), reshape(u, n, n)  # no need
        @debug "time = $t, trace = $(tr(mu))"
        lindblad_diffeq!(mdu, mu, evaluate(H, t), evaluate.(c_ops, t))
        e_ops_values = [expect(op, mu) for op in evaluate.(e_ops, t)]
        push!(times, t)
        push!(expect_values, e_ops_values)
        nothing
    end
    rho = dpsolve(f!, tspan, rho0; solver, integrator_config)
    return MesolveRes(times, expect_values, rho)
end
evaluate(op::AbstractMatrix, t) = op
evaluate(op::TimeDependent, t) = op.f(t)

# DormandPrince Solver with user defined input type
struct DPFunc{T, FT}
    f!::FT
    cache_u::T   # cache for input
    cache_du::T  # cache for gradient
end
function DPFunc(f!, u0)
    cache_u, cache_du = deepcopy(u0), deepcopy(u0)
    return DPFunc(f!, cache_u, cache_du)
end
function (f::DPFunc)(t, u::AbstractVector, du::AbstractVector)
    f.f!(t, setparams!(f.cache_u, u), setparams!(f.cache_du, du))
    getparams!(du, f.cache_du)
    nothing
end
function dpsolve(f!, tspan, u0::T; solver=:dp5, integrator_config=(;)) where T
    dpf = DPFunc(f!, u0)
    @assert solver ∈ (:dp5, :dp8) "unsupported solver $solver, should be `:dp5` or `:dp8`"
    SOLVER = solver === :dp5 ? DP5Solver : DP8Solver
    dpsolver = SOLVER(dpf, Float64(tspan[1]), copy(getparams(u0)); integrator_config...)
    DormandPrince.integrate!(dpsolver, Float64(tspan[2]))
    setparams!(dpf.cache_u, dpsolver.y)
    return dpf.cache_u
end
# support array inputs
getparams(x::AbstractArray{T}) where T = vec(x)
getparams!(y::AbstractVector, x::AbstractArray{T}) where T = y .= vec(x)
setparams!(y::AbstractArray{T}, x::AbstractVector) where T = y .= reshape(x, size(y))

struct Lindblad{TH, TC<:Tuple}
    H::TH
    c_ops::TC
end

function qihao_lindblad(;
            Omega_C = 6.20 * 2 *pi,            # Resonator Frequency (in angular frequency)
            Omega_1 = 5.50 * 2 * pi,           # Qubit Frequency (in angular frequency)
            J_C1 = 60 * 2 * pi/1000,           # JC Coupling Strength Qubit and Resonator
            Alpha_1 = -247 * 2 * pi/1000,      # Anharmonicity of Qubit
            # Chi = 0.4 * 2 * pi/1000         # Dispersive Shift (Roughly equals J_C1**2/Delta)
            Gamma_1 = 0.05 /1000,             # Gamma_1, Qubit Decoherence Rate (20.0 us)
            Gamma_phi = 0.5 /1000,            # Gamma_phi, Qubit Dephasing Rate (2.0 us)
            Amp = 0.080659, # A Low Readout Power
            Q_C = 60,         # Trancute the Cavity Hilbert Space
            Q_1 = 5,         # Trancute the Qubit Hilbert Space
            Freq = 6.20211134 * (2*pi), # Dispersive Readout Frequency
        )          # Kappa, Resonator Decay Rate
    Delta = abs(Omega_C - Omega_1)     # Detuning bewteen Qubit and Resonator
    Kappa = 2* Omega_C/7500
    K_1 = Alpha_1 * (J_C1/Delta)^4  # Kerr Self-interaction for the Dispersive Model
    Chi_1 = 2 * J_C1^2* Alpha_1/(Delta*(Delta+Alpha_1)) # Analytical Dispersive Shift

    # Define the Operators
    a = kron(destroy(Q_C), qeye(Q_1))    # Cavity Destroy Operator
    b = kron(qeye(Q_C), destroy(Q_1))    # Qubit Destroy Operator

    Proj_00 = kron(qeye(Q_C), SparseMatrixCSC(basis(Q_1,0) |> projector))
    Proj_01 = kron(qeye(Q_C), SparseMatrixCSC(basis(Q_1,1) |> projector))
    Proj_02 = kron(qeye(Q_C), SparseMatrixCSC(basis(Q_1,2) |> projector))

    Omega_C_tilde = 1/2 * (Omega_C + Omega_1 + sqrt(Delta^2 + 4 * J_C1^2))
    Omega_1_tilde = 1/2 * (Omega_C + Omega_1 - sqrt(Delta^2 + 4 * J_C1^2))

    # The Collapse Operators
    c_ops = Any[sqrt(Kappa) * a,
            sqrt(Gamma_1) * b,
            sqrt(Gamma_phi/2)* b' * b]
    # The Expectation Operators
    e_ops = Any[a' .+ a, -1im .* (a' .- a) , a' * a, Proj_00, Proj_01, Proj_02]

    H_0 =  (Omega_C_tilde-Freq) * a' * a + (Omega_1_tilde-Freq) * b' * b +
            Alpha_1/2 * b' * b' * b * b + K_1/2 * a' * a' * a * a +
            Chi_1 * a' * a * b' *b
    H_D = Amp * (a' + a)
    H = H_0 + H_D
    return Lindblad(H, (c_ops...,)), e_ops
end



function mesolve(state::AbstractVector, lin::Lindblad;
            e_ops,
            T_tot = 2000.0,                      # Total Time for Readout(ns)
            integrator_config=(;)
        )          # Kappa, Resonator Decay Rate
    return mesolve(lin.H, state * state', (0.0, T_tot); c_ops=lin.c_ops, e_ops, integrator_config)
end

