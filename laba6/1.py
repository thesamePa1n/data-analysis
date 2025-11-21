import math

yt = [27.553, 28.071, 28.162, 28.643, 29.636, 29.876, 30.015, 30.619,
      31.656, 31.822, 31.956, 32.712]
t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
n = 12
sum_t = sum(t)
sum_square_t = sum([i**2 for i in t])
sum_t_yt = sum([i * j for i, j in zip(yt, t)])
sum_y = sum(yt)

y_wave = [27.45, 27.92, 28.39, 28.86, 29.33, 29.8, 30.27, 30.74, 31.21, 31.68, 32.15, 32.62]
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
k = 8
t_avg = sum(t) / n
S_e = math.sqrt((1/(n-p))*sum([i**2 for i in e_t]))
t_alpha = 2.228
U_k = S_e * t_alpha * math.sqrt(1 + 1/n + (n + k - t_avg)**2 / sum([(i - t_avg)**2 for i in t]))
print(U_k)