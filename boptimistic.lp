\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[0] - 10 y[0]
Subject To
 R0: x[0] - 4 y[0] >= -11
 R1: - x[0] - 2 y[0] >= -13
 R2: 2 x[0] + y[0] >= 7
 R3: gamma[0] <= 1
 R4: gamma[0] - 50 alpha[0] <= 0
 R5: 2 x[0] + y[0] + 50 alpha[0] <= 57
 R6: y[0] - 50 beta[0] <= 0
 R7: gamma[0] - 50 beta[0] >= -49
Bounds
Binaries
 alpha[0] beta[0]
End
