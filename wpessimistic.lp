\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[0] - 10 ybar[0]
Subject To
 R0: x[0] - 4 ybar[0] >= -11
 R1: - x[0] - 2 ybar[0] >= -13
 R2: 2 x[0] + ybar[0] >= 3
 R3: - y[0] + ybar[0] <= 0
 R4: - pi1[0] + pi2[0] >= -10
 R5: pi1[0] - 50 upi1[0] <= 0
 R6: 2 x[0] + ybar[0] + 50 upi1[0] <= 53
 R7: pi2[0] - 50 upi2[0] <= 0
 R8: - y[0] + ybar[0] - 50 upi2[0] >= -50
 R9: ybar[0] - 50 uybar[0] <= 0
 R10: - pi1[0] + pi2[0] + 50 uybar[0] <= 50
 R11: 2 x[0] + y[0] >= 3
 R12: gamma[0] <= 1
 R13: gamma[0] - 50 alpha[0] <= 0
 R14: 2 x[0] + y[0] + 50 alpha[0] <= 53
 R15: y[0] - 50 beta[0] <= 0
 R16: gamma[0] - 50 beta[0] >= -49
Bounds
Binaries
 upi2[0] alpha[0] upi1[0] beta[0] uybar[0]
End
