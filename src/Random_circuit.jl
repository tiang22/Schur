using SparseArrays
using LinearAlgebra

include("mylindblad.jl")

using PyCall
function transinTwo(total_length, number)
	binary_string = string(number, base = 2)
	padded_string = lpad(binary_string, total_length, '0')
	return padded_string
end

function middle_test(oper_mat)
	initial_state = sparse([2^(2) + 2^(3) + 1], [1], [1], 2^(4 + 3 + 2), 1)
	final_state = oper_mat * initial_state
	final_state = Matrix(final_state)
	for nz in range(1, size(final_state)[1])
		if final_state[nz][1] != 0
			println(nz - 1, " ", transinTwo(4 + 3 + 2, nz - 1), " ", final_state[nz][1])
		end
	end
end

py"""
import numpy as np

def vec_pos(j1, j2, m1, m2):
	id1 = (j1 - m1)
	id2 = (j2 - m2)
	return int(id1 * (2 * j2 + 1) + id2)

def J_pos(Jmax, Ji, Mi):
	'''
	Ji in range(Jmin, Jmin+1, ..., Jmax)
	'''
	return int((Ji - Mi) + ((2 * Jmax + 1) + (2 * (Ji+1) + 1)) * (Jmax - Ji) / 2)

def uni_array(start, end, step):
	return np.arange(start, end + step, step)

def CG_Coef(j1, j2):
	'''
	计算(j1, [-j1,-j1+1, ... ,j1] ) tensor (j2, [-j2, -j2+1, ... , j2] ) 空间的 J invariant 子空间直和形式
	即( j1-j2, spin-z ), ( j1 - j2 + 1, spin-z, ), ... , ( j1 + j2, spin-z )子空间的形式
	assume j1 >= j2
	维度: 
	(2j1+1)*(2j2+1) = [ 2(j1 - j2) +1 ] + [ 2(j1-j2+1) + 1] + ... + [ 2(j1 + j2) +1 ]

	认为 (j1, s1) tensor (j2, s2) 占据的行是 s1 * (2j2+1) + s2 (input Vector)
	'''

	mat = np.zeros([(int)(2 * j1 + 1) * (int)(2 * j2 + 1),
				   (int)(2 * j1 + 1) * (int)(2 * j2 + 1)])

	row = 0

	for J in uni_array(j1 + j2, np.abs(j1 - j2), -1):
		# 单独计算每个J的最大的M所对应的变换系数
		if (J == (j1 + j2)):
			mat[0][vec_pos(j1, j2, j1, j2)] = 1
		else:
			is_first_m1 = False
			for m1 in uni_array(j1, -j1, -1):
				m2 = J - m1
				if (m2 > j2 or m2 < - j2):
					continue
				if (is_first_m1 == False):  # 确定第一个m1
					sum = 0
					for uper_J in uni_array(m1 + m2 + 1, j1 + j2, 1):
						sum += mat[J_pos(j1 + j2, uper_J, m1 + m2)
								   ][vec_pos(j1, j2, m1, m2)] ** 2
					# (J, J)对应的(m1, m2)应该与其他向量组正交
					mat[J_pos(j1 + j2, J, J)][vec_pos(j1, j2, m1, m2)
											  ] = np.sqrt(1 - sum)
					is_first_m1 = True
				else:
					sum = 0
					for uper_J in uni_array(m1 + m2 + 1, j1 + j2, 1):
						sum += mat[J_pos(j1 + j2, uper_J, m1 + m2)][vec_pos(j1, j2, m1, m2)] * \
							mat[J_pos(j1 + j2, uper_J, m1+m2)
								][vec_pos(j1, j2, m1+1, m2-1)]

					mat[J_pos(j1 + j2, J, J)][vec_pos(j1, j2, m1, m2)] = (0 - sum) / \
						mat[J_pos(j1 + j2, J, J)][vec_pos(j1, j2, m1+1, m2-1)]

		for M in uni_array(J, - J + 1, -1):  # 作用降算子
			for m1 in uni_array(j1, -j1, -1):
				m2 = M - m1 - 1
				if (m2 > j2 or m2 < - j2):
					continue
				mat[J_pos(j1 + j2, J, M - 1)][vec_pos(j1, j2, m1, m2)] += mat[J_pos(j1 + j2,
																					J, M)][vec_pos(j1, j2, m1+1, m2)] * np.sqrt((j1 + 1 + m1) * (j1 - m1))
				mat[J_pos(j1 + j2, J, M - 1)][vec_pos(j1, j2, m1, m2)] += mat[J_pos(j1 + j2,
																					J, M)][vec_pos(j1, j2, m1, m2+1)] * np.sqrt((j2 + 1 + m2) * (j2 - m2))
				mat[J_pos(j1 + j2, J, M - 1)][vec_pos(j1, j2, m1, m2)
											  ] /= np.sqrt((J + 1 - M) * (J + M))

	return mat

if __name__ == "__main__":
	mat = CG_Coef(1, 1/2)
	print(mat.round(decimals=2))

"""

function CG_Coef(j1, j2)
	mat = py"CG_Coef"(j1, j2)
	return mat
end


function sparse_identity(n)
	return spdiagm(0 => ones(ComplexF64, 2^n))
end

# write an algorithm to count the number of bits of a number
function count_bits(n)
	count = 0
	if n == 0
		return 1
	end
	while n > 0
		count += 1
		n = n >> 1
	end
	return count
end

# write an algorithm to generate the CG matrix with the given twoJ, nSTAT, nP, usedSTAT, nowP
function CG_matrix(twoJ, nP, nSTAT, usedSTAT, nowP)

	real_input_vec_position = []
	for freequbits in 0:0
		for i in 0:twoJ
			push!(real_input_vec_position, 2^(nP - nowP + nSTAT) * 0 + freequbits * 2^(nSTAT) + i * 2^(nSTAT - usedSTAT)) # 0 0...0 (state)
			push!(real_input_vec_position, 2^(nP - nowP + nSTAT) * 1 + freequbits * 2^(nSTAT) + i * 2^(nSTAT - usedSTAT)) # 1 0...0 (state)
		end
	end
	# calculate the number of used state qubits for Jplus space, minus one because of n->log2(n-1)
	Jplusqubits = count_bits(twoJ + 2 - 1)
	Jminusqubits = count_bits(twoJ - 1)
	real_output_vec_position = []
	for freequbits in 0:0
		for i in 0:(twoJ+1)
			push!(real_output_vec_position, 2^(nP - nowP + nSTAT) * 0 + freequbits * 2^(nSTAT) + i * 2^(nSTAT - Jplusqubits))
		end
		# calculate the number of used state qubits for Jminus space, minus one because of n->log2(n-1)
		for i in 0:(twoJ-1)
			push!(real_output_vec_position, 2^(nP - nowP + nSTAT) * 1 + freequbits * 2^(nSTAT) + i * 2^(nSTAT - Jminusqubits))
		end
	end


	# println(real_output_vec_position)
	# println(real_input_vec_position)
	mat = CG_Coef(twoJ / 2, 1 / 2)
	# println(mat' * mat)


	ret_mat = sparse_identity(nP - nowP + 1 + nSTAT)

	for i in 1:length(real_output_vec_position)
		ret_mat[real_output_vec_position[i]+1, real_output_vec_position[i]+1] = 0
		ret_mat[real_input_vec_position[i]+1, real_input_vec_position[i]+1] = 0
	end

	for i in 1:length(real_output_vec_position)
		for j in 1:length(real_input_vec_position)
			ret_mat[real_output_vec_position[i]+1, real_input_vec_position[j]+1] = mat[(i-1)%(2*twoJ+2)+1, (j-1)%(2*twoJ+2)+1]
		end
	end

	common = intersect(real_output_vec_position, real_input_vec_position)
	filter_output = setdiff(real_output_vec_position, common)
	filter_input = setdiff(real_input_vec_position, common)
	for i in 1:length(filter_output)
		ret_mat[filter_input[i]+1, filter_output[i]+1] = 1
	end

	for freequbits in 1:(2^(nP-nowP)-1)
		for i in 1:length(real_output_vec_position)
			ret_mat[real_output_vec_position[i]+freequbits*2^(nSTAT)+1, real_output_vec_position[i]+freequbits*2^(nSTAT)+1] = 0
			ret_mat[real_input_vec_position[i]+freequbits*2^(nSTAT)+1, real_input_vec_position[i]+freequbits*2^(nSTAT)+1] = 0
		end

		for i in 1:length(real_output_vec_position)
			for j in 1:length(real_input_vec_position)
				ret_mat[real_output_vec_position[i]+freequbits*2^(nSTAT)+1, real_input_vec_position[j]+freequbits*2^(nSTAT)+1] = mat[i, j]
			end

			for i in 1:length(filter_output)
				ret_mat[filter_input[i]+freequbits*2^(nSTAT)+1, filter_output[i]+freequbits*2^(nSTAT)+1] = 1
			end
		end
	end

	return ret_mat
end

function control_CG_transform(twoJ, nSP, nP, nSTAT, usedSTAT, nowP)  # spin 2J is qubit (2J+1)
	U = kron(sparse_identity(nowP - 1), CG_matrix(twoJ, nP, nSTAT, usedSTAT, nowP))
	I = sparse_identity(nP + nSTAT)

	control_0_mat = sparse([1], [1], [1], 2, 2)
	control_1_mat = sparse([2], [2], [1], 2, 2)
	pre_identity_SP = sparse_identity(twoJ) # 注意spin-J 前面有 2J个qubit
	suf_identity_SP = sparse_identity(nSP - (twoJ + 1)) # 后面有 nSP - 2J -1 个qubit

	control_0_mat = kron(kron(pre_identity_SP, control_0_mat), suf_identity_SP)
	control_1_mat = kron(kron(pre_identity_SP, control_1_mat), suf_identity_SP)

	ret_mat = kron(control_0_mat, I) + kron(control_1_mat, U)
	return ret_mat
end

function noise_time_evolution(nSP, nP, nSTAT, rho, delta_t)
	# return rho # no noise firstly

	# noise model
	C = sqrt(100000) * sum([kron(kron(sparse_identity(i - 1), sparse([1, 2], [1, 2], [1, -1], 2, 2)), sparse_identity(nSP + nP + nSTAT - i)) for i in 1:nSP+nP+nSTAT])
	H = sparse_identity(nSP + nP + nSTAT)
	tspan = (0.0, delta_t)

	return Lindblad_equations(rho, H, C, tspan)
end


# the first step of control swap
function control_swap_1(nSP, nP, nSTAT, ctrla, octrlb, oper)
	pre_mat = sparse_identity(ctrla - 1)
	oper_mat1 = kron(kron(sparse([1], [1], [1], 2, 2), sparse_identity(octrlb - oper)), sparse([1], [1], [1], 2, 2))
	oper_mat2 = kron(kron(kron(sparse([2], [2], [1], 2, 2), sparse([1, 2], [2, 1], [1, 1], 2, 2)), sparse_identity(octrlb - oper - 1)), sparse([1], [1], [1], 2, 2))
	oper_mat3 = kron(kron(sparse([1], [1], [1], 2, 2), sparse_identity(octrlb - oper)), sparse([2], [2], [1], 2, 2))
	oper_mat4 = kron(kron(sparse([2], [2], [1], 2, 2), sparse_identity(octrlb - oper)), sparse([2], [2], [1], 2, 2))
	suf_mat = sparse_identity(nSP + nP + nSTAT - octrlb)
	opermat = oper_mat1 + oper_mat2 + oper_mat3 + oper_mat4
	return kron(kron(pre_mat, opermat), suf_mat)
end

# the second step of control swap
function control_swap_2(nSP, nP, nSTAT, ctrla, ctrlb, oper)
	pre_mat = sparse_identity(ctrla - 2)
	oper_mat1 = kron(kron(kron(sparse_identity(1), sparse([1], [1], [1], 2, 2)), sparse_identity(ctrlb - ctrla - 1)), sparse([1], [1], [1], 2, 2))
	oper_mat2 = kron(kron(kron(sparse_identity(1), sparse([2], [2], [1], 2, 2)), sparse_identity(ctrlb - ctrla - 1)), sparse([1], [1], [1], 2, 2))
	oper_mat3 = kron(kron(kron(sparse_identity(1), sparse([1], [1], [1], 2, 2)), sparse_identity(ctrlb - ctrla - 1)), sparse([2], [2], [1], 2, 2))
	oper_mat4 = kron(kron(kron(sparse([1, 2], [2, 1], [1, 1], 2, 2), sparse([2], [2], [1], 2, 2)), sparse_identity(ctrlb - ctrla - 1)), sparse([2], [2], [1], 2, 2))
	suf_mat = sparse_identity(nSP + nP + nSTAT - ctrlb)
	opermat = oper_mat1 + oper_mat2 + oper_mat3 + oper_mat4
	return kron(kron(pre_mat, opermat), suf_mat)
end

# the third step of control swap
function control_swap_3(nSP, nP, nSTAT, ctrla, ctrlb, oper)
	pre_mat = sparse_identity(ctrla - 1)
	oper_mat1 = kron(kron(kron(sparse([1], [1], [1], 2, 2), sparse_identity(1)), sparse_identity(ctrlb - oper - 1)), sparse([1], [1], [1], 2, 2))
	oper_mat2 = kron(kron(kron(sparse([2], [2], [1], 2, 2), sparse_identity(1)), sparse_identity(ctrlb - oper - 1)), sparse([1], [1], [1], 2, 2))
	oper_mat3 = kron(kron(kron(sparse([1], [1], [1], 2, 2), sparse_identity(1)), sparse_identity(ctrlb - oper - 1)), sparse([2], [2], [1], 2, 2))
	oper_mat4 = kron(kron(kron(sparse([2], [2], [1], 2, 2), sparse([1, 2], [2, 1], [1, 1], 2, 2)), sparse_identity(ctrlb - oper - 1)), sparse([2], [2], [1], 2, 2))
	suf_mat = sparse_identity(nSP + nP + nSTAT - ctrlb)
	opermat = oper_mat1 + oper_mat2 + oper_mat3 + oper_mat4
	return kron(kron(pre_mat, opermat), suf_mat)
end

# the fourth step of control swap
function control_swap_4(nSP, nP, nSTAT, ctrla, octrlb, oper)
	pre_mat = sparse_identity(ctrla - 2)
	oper_mat1 = kron(kron(kron(sparse_identity(1), sparse([1], [1], [1], 2, 2)), sparse_identity(octrlb - ctrla - 1)), sparse([1], [1], [1], 2, 2))
	oper_mat2 = kron(kron(kron(sparse_identity(1), sparse([1], [1], [1], 2, 2)), sparse_identity(octrlb - ctrla - 1)), sparse([2], [2], [1], 2, 2))
	oper_mat3 = kron(kron(kron(sparse_identity(1), sparse([2], [2], [1], 2, 2)), sparse_identity(octrlb - ctrla - 1)), sparse([2], [2], [1], 2, 2))
	oper_mat4 = kron(kron(kron(sparse([1, 2], [2, 1], [1, 1], 2, 2), sparse([2], [2], [1], 2, 2)), sparse_identity(octrlb - ctrla - 1)), sparse([1], [1], [1], 2, 2))
	suf_mat = sparse_identity(nSP + nP + nSTAT - octrlb)
	opermat = oper_mat1 + oper_mat2 + oper_mat3 + oper_mat4
	return kron(kron(pre_mat, opermat), suf_mat)
end

function FirstTransform(nSP, nP, nSTAT, first_qubit) # 原本的态存储在first_qubit上
	twoJ = 1
	state = nSP + nP + 1
	ret_mat = kron(kron(sparse_identity(1), sparse([1, 2], [2, 1], [1, 1], 2, 2)), sparse_identity(nSP + nP + nSTAT - 2)) # 把 所有态都先标记为 2J = 1 的态

	oper_mat1 = kron(kron(sparse_identity(first_qubit - 1), sparse([1], [1], [1], 2, 2)), sparse_identity(nSP + nP + nSTAT - first_qubit))
	oper_mat2 = kron(kron(kron(kron(sparse_identity(first_qubit - 1), sparse([2], [2], [1], 2, 2)), sparse_identity(state - first_qubit - 1)), sparse([1, 2], [2, 1], [1, 1], 2, 2)), sparse_identity(nSP + nP + nSTAT - state))
	oper_mat_1 = oper_mat1 + oper_mat2

	oper_mat1 = kron(kron(sparse_identity(state - 1), sparse([1], [1], [1], 2, 2)), sparse_identity(nSP + nP + nSTAT - state))
	oper_mat2 = kron(kron(kron(kron(sparse_identity(first_qubit - 1), sparse([1, 2], [2, 1], [1, 1], 2, 2)), sparse_identity(state - first_qubit - 1)), sparse([2], [2], [1], 2, 2)), sparse_identity(nSP + nP + nSTAT - state))
	oper_mat_2 = oper_mat1 + oper_mat2
	return oper_mat_2 * oper_mat_1 * ret_mat
end

function Schur_Transform_Random(n, input_state, delta_t)
	nSP = n + 1
	nP = n
	nSTAT = convert(Int64, ceil(log2(n + 1)))

	dm = kron(input_state, input_state')

	oper_mat = FirstTransform(nSP, nP, nSTAT, nSP + 1)
	dm = oper_mat * dm * oper_mat'
	println("Yes")
	dm = noise_time_evolution(nSP, nP, nSTAT, dm, delta_t)
	for Time in 1:n-1
		# calculate which J is vaild
		println("Fixed input", " ", Time)
		nowJ = []
		if Time & 1 == 1
			for i in 1:2:Time
				push!(nowJ, i)
			end
		else
			for i in 0:2:Time
				push!(nowJ, i)
			end
		end

		for twoJ in nowJ
			oper_mat = control_CG_transform(twoJ, nSP, nP, nSTAT, count_bits(twoJ), Time + 1) # used 2J + 1 states, means log2(2J+1 - 1) qubits
			dm = oper_mat * dm * oper_mat'
			dm = noise_time_evolution(nSP, nP, nSTAT, dm, delta_t)
		end
		println("Fixed control transform", " ", Time)

		for twoJ in nowJ
			oper_mat = control_swap_1(nSP, nP, nSTAT, twoJ + 1, nSP + Time + 1, twoJ + 2) # 第 (2J+1) 个qubit
			dm = oper_mat * dm * oper_mat'
			dm = noise_time_evolution(nSP, nP, nSTAT, dm, delta_t)
			if twoJ != 0
				oper_mat = control_swap_2(nSP, nP, nSTAT, twoJ + 1, nSP + Time + 1, twoJ)
				dm = oper_mat * dm * oper_mat'
				dm = noise_time_evolution(nSP, nP, nSTAT, dm, delta_t)
			end
			if twoJ != 0
				oper_mat = control_swap_3(nSP, nP, nSTAT, twoJ, nSP + Time + 1, twoJ + 1)
				dm = oper_mat * dm * oper_mat'
				dm = noise_time_evolution(nSP, nP, nSTAT, dm, delta_t)
			end
			oper_mat = control_swap_4(nSP, nP, nSTAT, twoJ + 2, nSP + Time + 1, twoJ + 1)
			dm = oper_mat * dm * oper_mat'
			dm = noise_time_evolution(nSP, nP, nSTAT, dm, delta_t)
		end
	end
	return dm
end

function vaild_check(n, J, mu, nP)
	if (n + J) % 2 != 0
		return false
	end
	upper = (n + J) / 2
	lower = (n - J) / 2
	numbers = transinTwo(nP, mu)
	s_upper = 0
	s_lower = 0
	for i in 1:nP
		if numbers[i] == '1'
			s_lower += 1
		else
			s_upper += 1
		end
		if s_lower > s_upper
			return false
		end
	end
	if s_upper != upper || s_lower != lower
		return false
	end
	return true
end

function MatrixShow(n, mat)
	nSP = n + 1
	nP = n
	nSTAT = convert(Int64, ceil(log2(n + 1)))
	answer = zeros(2^n, 2^n)
	for initial_state in 0:2^(n)-1
		real_initial_state = initial_state * 2^(nSTAT)
		input_state = sparse([real_initial_state + 1], [1], [1], 2^(nSP + nP + nSTAT), 1)
		output_state = mat * input_state
		output_state = Matrix(output_state)
		cnt = 0
		for J in (n):-1:0
			for mu in 0:2^(nP)-1
				if vaild_check(n, J, mu, nP)
					for ms in 0:(J)
						used_qubits = count_bits(J)
						real_output_state = 2^(nP + nSTAT + n - J) + mu * 2^(nSTAT) + ms * 2^(nSTAT - used_qubits)
						if output_state[real_output_state+1][1] != 0
							answer[cnt+1, initial_state+1] = output_state[real_output_state+1][1]
						end
						cnt = cnt + 1
					end
				end
			end
		end
	end

	rounded_answer = round.(answer, digits = 3)
	for row in eachrow(rounded_answer)
		for element in row
			print(element, " ")
		end
		println()
	end
end


N = 2
nSP = N + 1
nP = N
nSTAT = convert(Int64, ceil(log2(N + 1)))
input_state = sparse([0 + 1, 2^nSTAT + 1], [1, 1], [1 / sqrt(2), 1 / sqrt(2)], 2^(nSP + nP + nSTAT), 1)
# input_state = sparse([0 + 1], [1], [1], 2^(nSP + nP + nSTAT), 1)   

oper_mat = Schur_Transfor_random(N)
sample_output_state = oper_mat * input_state
sample_output_dm = kron(sample_output_state, sample_output_state')

test_output_dm = Noisy_Schur_Transform(N, input_state, 0.03)


delta_dm = test_output_dm - sample_output_dm
for i in 1:size(delta_dm)[1]
	for j in 1:size(delta_dm)[2]
		if delta_dm[i, j] != 0
			println("i= ", i, " j= ", j, " value= ", delta_dm[i, j])
		end
	end
end

l2_norm = norm(delta_dm, 2)
