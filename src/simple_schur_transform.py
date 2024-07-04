import numpy as np


def vec_pos(j1, j2, m1, m2):
    id1 = j1 - m1
    id2 = j2 - m2
    return int(id1 * (2 * j2 + 1) + id2)


def J_pos(Jmax, Ji, Mi):
    """
    Ji in range(Jmin, Jmin+1, ..., Jmax)
    """
    return int((Ji - Mi) + ((2 * Jmax + 1) + (2 * (Ji + 1) + 1)) * (Jmax - Ji) / 2)


def uni_array(start, end, step):
    return np.arange(start, end + step, step)


def CG_Coef(j1, j2):
    """
    计算(j1, [-j1,-j1+1, ... ,j1] ) tensor (j2, [-j2, -j2+1, ... , j2] ) 空间的 J invariant 子空间直和形式
    即( j1-j2, spin-z ), ( j1 - j2 + 1, spin-z, ), ... , ( j1 + j2, spin-z )子空间的形式
    assume j1 >= j2
    维度:
    (2j1+1)*(2j2+1) = [ 2(j1 - j2) +1 ] + [ 2(j1-j2+1) + 1] + ... + [ 2(j1 + j2) +1 ]

    认为 (j1, s1) tensor (j2, s2) 占据的列向量下标是 s1 * (2j2+1) + s2
    """

    mat = np.zeros(
        [(int)(2 * j1 + 1) * (int)(2 * j2 + 1), (int)(2 * j1 + 1) * (int)(2 * j2 + 1)]
    )

    row = 0

    for J in uni_array(j1 + j2, np.abs(j1 - j2), -1):
        # 单独计算每个J的最大的M所对应的变换系数
        if J == (j1 + j2):
            mat[0][vec_pos(j1, j2, j1, j2)] = 1
        else:
            is_first_m1 = False
            for m1 in uni_array(j1, -j1, -1):
                m2 = J - m1
                if m2 > j2 or m2 < -j2:
                    continue
                if is_first_m1 == False:  # 确定第一个m1
                    sum = 0
                    for uper_J in uni_array(m1 + m2 + 1, j1 + j2, 1):
                        sum += (
                            mat[J_pos(j1 + j2, uper_J, m1 + m2)][
                                vec_pos(j1, j2, m1, m2)
                            ]
                            ** 2
                        )
                    # (J, J)对应的(m1, m2)应该与其他向量组正交
                    mat[J_pos(j1 + j2, J, J)][vec_pos(j1, j2, m1, m2)] = np.sqrt(
                        1 - sum
                    )
                    is_first_m1 = True
                else:
                    sum = 0
                    for uper_J in uni_array(m1 + m2 + 1, j1 + j2, 1):
                        sum += (
                            mat[J_pos(j1 + j2, uper_J, m1 + m2)][
                                vec_pos(j1, j2, m1, m2)
                            ]
                            * mat[J_pos(j1 + j2, uper_J, m1 + m2)][
                                vec_pos(j1, j2, m1 + 1, m2 - 1)
                            ]
                        )

                    mat[J_pos(j1 + j2, J, J)][vec_pos(j1, j2, m1, m2)] = (
                        0 - sum
                    ) / mat[J_pos(j1 + j2, J, J)][vec_pos(j1, j2, m1 + 1, m2 - 1)]

        for M in uni_array(J, -J + 1, -1):  # 作用降算子
            for m1 in uni_array(j1, -j1, -1):
                m2 = M - m1 - 1
                if m2 > j2 or m2 < -j2:
                    continue
                mat[J_pos(j1 + j2, J, M - 1)][vec_pos(j1, j2, m1, m2)] += mat[
                    J_pos(j1 + j2, J, M)
                ][vec_pos(j1, j2, m1 + 1, m2)] * np.sqrt((j1 + 1 + m1) * (j1 - m1))
                mat[J_pos(j1 + j2, J, M - 1)][vec_pos(j1, j2, m1, m2)] += mat[
                    J_pos(j1 + j2, J, M)
                ][vec_pos(j1, j2, m1, m2 + 1)] * np.sqrt((j2 + 1 + m2) * (j2 - m2))
                mat[J_pos(j1 + j2, J, M - 1)][vec_pos(j1, j2, m1, m2)] /= np.sqrt(
                    (J + 1 - M) * (J + M)
                )

    return mat


if __name__ == "__main__":
    mat = CG_Coef(1, 1 / 2)
    print(mat.round(decimals=2))
