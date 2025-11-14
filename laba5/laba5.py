import math

yt = [27.553, 28.071, 28.162, 28.643, 29.636, 29.876, 30.015, 30.619,
      31.656, 31.822, 31.956, 32.712]

yt_avg = sum(yt) / 12

diff_yt = [0.518, 0.091, 0.481, 0.993, 0.24, 0.139, 0.604, 1.037, 0.166, 0.134, 0.756]
Sy = math.sqrt(sum([(i - yt_avg)**2 for i in yt]) / 11)
lmda_t = [i / Sy for i in diff_yt]
print(lmda_t)