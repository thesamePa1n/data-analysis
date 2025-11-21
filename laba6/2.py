import math

yt = [31, 52, 61, 73, 81, 91, 96, 99, 107, 113, 117, 123, 133, 137, 149, 163]
t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
n = 16
sum_t = sum(t)
sum_square_t = sum([i**2 for i in t])
sum_t_yt = sum([i * j for i, j in zip(yt, t)])
sum_y = sum(yt)

y_wave = [37.875 + 7.5*i for i in t]
e_t = [i - j for i, j in zip(yt, y_wave)]
e_avg = sum(e_t) / n
S = math.sqrt(sum([(i - e_avg)**2 for i in e_t]) / (n - 1))
t_pach = abs(e_avg) * math.sqrt(n) / S

dw = sum([(e_t[i] - e_t[i-1])**2 for i in range(1, len(e_t))]) / sum([i**2 for i in e_t])

As = ((1/n)*sum([i**3 for i in e_t])) / math.sqrt((1/n)*(sum([i**2 for i in e_t]))**3)
Ex = (sum([i**4 for i in e_t]) / math.sqrt(sum([i**2 for i in e_t])**2)) - 3

sigma_As = math.sqrt(6*(n-2)*(n+1)*(n+3))
sigma_Ex = math.sqrt((24*n*(n-2)*(n-3)) / ((n+1)**2*(n+3)*(n+5)))

RS = (max(e_t) - min(e_t)) / S

E = (1/n)*sum([abs(i)/j for i, j in zip(e_t, yt)]) * 100

p = 2
k = 6
t_avg = sum(t) / n
S_e = math.sqrt((1/(n-p))*sum([i**2 for i in e_t]))
t_alpha = 2.145
U_k = S_e * t_alpha * math.sqrt(1 + 1/n + (n + k - t_avg)**2 / sum([(i - t_avg)**2 for i in t]))

print(U_k)