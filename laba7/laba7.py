A = [33, 34, 37, 29, 31, 31, 29, 47]
B = [31, 31, 43, 38, 51, 35, 33, 35]
C = [27, 26, 29, 31, 33, 28, 30, 37]
n = 8
N = 24
c = 3

sum_A = sum(A)
sum_B = sum(B)
sum_C = sum(C)
total_sum = sum_A + sum_B + sum_C
avg_A = sum_A / n
avg_B = sum_B / n
avg_C = sum_C / n

Tc = [sum_A, sum_B, sum_C]
sum_square_Tc = sum([i**2 for i in Tc])
square_sum_xi = sum(Tc)**2
sum_square_xi = sum([i**2 for i in A]) + sum([i**2 for i in B]) + sum([i**2 for i in C])

SS_fact = (sum_square_Tc / n) - (square_sum_xi / N)
SS_common = sum_square_xi - square_sum_xi / N
SS_cl = SS_common - SS_fact

k_fact = 2
k_common = 23
k_cl = k_common - k_fact

MS_fact = SS_fact / k_fact
MS_cl = SS_cl / k_cl
F_emp = MS_fact / MS_cl