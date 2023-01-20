\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[0] - 10 y[0]
Subject To
 R0: x[0] - 4 y[0] >= -11
 R1: - x[0] - 2 y[0] >= -13
 R2: 2 x[0] + y[0] >= 5
 R3: - 5 x[0] + 4 y[0] >= -30
 R4: gamma[0] + 4 gamma[1] <= 1
 R5: gamma[0] - 50 alpha[0] <= 0
 R6: gamma[1] - 50 alpha[1] <= 0
 R7: 2 x[0] + y[0] + 50 alpha[0] <= 55
 R8: - 5 x[0] + 4 y[0] + 50 alpha[1] <= 20
 R9: y[0] - 50 beta[0] <= 0
 R10: gamma[0] + 4 gamma[1] - 50 beta[0] >= -49
Bounds
Binaries
 alpha[0] alpha[1] beta[0]
End
