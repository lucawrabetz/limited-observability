\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[0] - 10 y[0]
Subject To
 R0: x[0] + y[0] - 4 y[1] >= -11
 R1: - x[0] - y[0] - 2 y[1] >= -13
 R2: 2 x[0] + 2 y[0] + y[1] >= 5
 R3: - 5 x[0] - 5 y[0] + 4 y[1] >= -30
 R4: 2 lambda[0] - 5 lambda[1] <= 2
 R5: lambda[0] + 4 lambda[1] <= 1
 R6: lambda[0] - 1000 alpha[0] <= 0
 R7: lambda[0] + 1000 alpha[0] >= 0
 R8: lambda[1] - 1000 alpha[1] <= 0
 R9: lambda[1] + 1000 alpha[1] >= 0
 R10: 2 x[0] + 2 y[0] + y[1] + 1000 alpha[0] <= 1005
 R11: - 5 x[0] - 5 y[0] + 4 y[1] + 1000 alpha[1] <= 970
 R12: 2 x[0] + 2 y[0] + y[1] - 1000 alpha[0] >= -995
 R13: - 5 x[0] - 5 y[0] + 4 y[1] - 1000 alpha[1] >= -1030
 R14: x[0] = 0
Bounds
Binaries
 alpha[0] alpha[1]
End
