import numpy as np
from gurobipy import *

class ModelData:
    # Class to hold an instance of the nominal problem.
    # Default initializations are set to Example 1 in our paper.
    def __init__(self,
                  n_dim=1, # dimension of x variable
                  c=np.array([1]),
                  f=np.array([0]),
                  m_dim=2, # dimension of y variable
                  d=np.array([-10, 0]),
                  g=np.array([2, 1]),
                  k_dim=2, # number of upper level constraints
                  A=np.array([[1], [-1]]), # \in k \times n (upper level coefficients, x)
                  B=np.array([[1, -4], [-1, -2]]), # \in k \times m (upper level coefficients, y)
                  a=np.array([-11, -13]), # \in \in k \times 1 (upper level RHS)
                  l_dim=2, # number of lower level constraints
                  C=np.array([[2], [-5]]), # \in l \times n (lower level coefficients, x)
                  D=np.array([[2, 1], [-5, 4]]), # \in l \times m (lower level coefficients, y)
                  b=np.array([5, -30]), # \in \in l \times 1 (lower level RHS)
                  M=1000
                 ):
        # vector assertions
        assert c.shape[0] == n_dim, "c does not match n dimension"
        assert f.shape[0] == n_dim, "f does not match n dimension"
        assert c.ndim == 1, "c is not a vector"
        assert f.ndim == 1, "f is not a vector"
        assert d.shape[0] == m_dim, "d does not match m dimension"
        assert g.shape[0] == m_dim, "g does not match m dimension"
        assert d.ndim == 1, "d is not a vector"
        assert g.ndim == 1, "g is not a vector"
        assert a.shape[0] == k_dim, "a does not match k dimension"
        assert b.shape[0] == l_dim, "b does not match l dimension"
        assert a.ndim == 1, "a is not a vector"
        assert b.ndim == 1, "b is not a vector"
        # matrix assertions
        assert A.shape[0] == k_dim, "A rows don't match k dimension"
        assert A.shape[1] == n_dim, "A columns don't match n dimension"
        assert B.shape[0] == k_dim, "B rows don't match k dimension"
        assert B.shape[1] == m_dim, "B columns don't match m dimension"
        assert C.shape[0] == l_dim, "C rows don't match l dimension"
        assert C.shape[1] == n_dim, "C columns don't match n dimension"
        assert D.shape[0] == l_dim, "D rows don't match l dimension"
        assert D.shape[1] == m_dim, "D columns don't match m dimension"
        self.n_dim=n_dim
        self.c=c
        self.f=f
        self.m_dim=m_dim
        self.d=d
        self.g=g
        self.k_dim=k_dim
        self.A=A
        self.B=B
        self.a=a
        self.l_dim=l_dim
        self.C=C
        self.D=D
        self.b=b
        self.M=M

    def log_data(self):
        print("n:", self.n_dim)
        print("m:", self.m_dim)
        print("k:", self.k_dim)
        print("l:", self.l_dim)
        print("c:", self.c)
        print("d:", self.d)
        print("A:", self.A)
        print("B:", self.B)
        print("a:", self.a)
        print("f:", self.f)
        print("g:", self.g)
        print("C:", self.C)
        print("D:", self.D)
        print("b:", self.b)

def return_beck_example():
    # Return an instance matching Example 1 in the Beck paper.
    return ModelData(n_dim=1,
                  c=np.array([1]),
                  f=np.array([0]),
                  m_dim=1,
                  d=np.array([-10]),
                  g=np.array([1]),
                  k_dim=2,
                  A=np.array([[1], [-1]]),
                  B=np.array([[-4], [-2]]),
                  a=np.array([-11, -13]),
                  l_dim=2,
                  C=np.array([[2], [-5]]),
                  D=np.array([[1], [4]]),
                  b=np.array([5, -30]),
                  M=50)

def print_solution(model, data, x, y, dual=None):
    if model.Status == GRB.Status.OPTIMAL:
        print("x:")
        for i in range(data.n_dim):
            print("   ", i, ":", x[i].x)
        print("y:")
        for i in range(data.m_dim):
            print("   ", i, ":", y[i].x)
        if dual:
            print("dual:")
            for i in range(data.l_dim):
                print("   ", i, ":", dual[i].x)

def construct_base_model(model, data):
    # Construct the part of the model that is common to all formulations/problems.
    # Add variables x and y, add objective function and primal upper constraint.
    x = model.addMVar(data.n_dim, name="x")
    y = model.addMVar(data.m_dim, name="y")
    # Set objective function
    model.setObjective(data.c @ x + data.d @ y, GRB.MINIMIZE)
    # Add constraint
    model.addConstr(data.A @ x + data.B @ y >= data.a)
    return x, y

def solve_nominal(data):
    # Constructed assuming x, y have non-negativity constraints.
    # Initialize Model
    nominal=Model()
    x, y = construct_base_model(nominal, data)
    # Add variables gamma, alpha, beta
    gamma = nominal.addMVar(data.l_dim, name="gamma")
    alpha = nominal.addMVar(data.l_dim, vtype=GRB.BINARY, name="alpha")
    beta = nominal.addMVar(data.m_dim, vtype=GRB.BINARY, name="beta")
    nominal.addConstr(data.C @ x + data.D @ y >= data.b)
    nominal.addConstr(data.D.transpose() @ gamma <= data.g)
    # complimentary slackness - gamma[i] == 0 if alpha[i] == 0 added individually
    for i in range(data.l_dim):
        nominal.addConstr(gamma[i] <= data.M * alpha[i])
    slack_expr = data.C @ x + data.D @ y - data.b
    M_mat = np.eye(data.l_dim) * data.M
    M1 = M_mat @ np.ones(data.l_dim) - M_mat @ alpha
    nominal.addConstr(slack_expr <= M1)
    # complimentary slackness - y[j] == 0 if beta[j] == 0 added individually
    for j in range(data.m_dim):
        nominal.addConstr(y[j] <= data.M * beta[j])
    slack_expr = np.transpose(data.D) @ gamma - data.g
    M_mat = np.eye(data.m_dim) * data.M
    M2 = -M_mat @ np.ones(data.m_dim) + M_mat @ beta
    nominal.addConstr(slack_expr >= M2)
    nominal.update()
    nominal.write("nominal.lp")
    nominal.optimize()
    print_solution(nominal, data, x, y)

def solve_beck_optimistic(model_data, P):
    # Constructed assuming x, y have non-negativity constraints.
    # Initialize Model
    boptimistic=Model()
    x, y = construct_base_model(boptimistic, data)
    # Add variables lambda, alpha
    lam = boptimistic.addMVar(data.l_dim, name="lambda")
    alpha = boptimistic.addMVar(data.l_dim, vtype=GRB.BINARY, name="alpha")
    boptimistic.addConstr(data.C @ x + data.D @ y >= data.b)
    boptimistic.addConstr(data.D.transpose() @ lam <= data.g)
    # complimentary slackness - lam[i] == 0 if alpha[i] == 0 added individually
    for i in range(data.l_dim):
        boptimistic.addConstr(lam[i] <= data.M * alpha[i])
        boptimistic.addConstr(lam[i] >= -data.M * alpha[i])
    slack_expr = data.C @ x + data.D @ y - data.b
    M_mat = np.eye(data.l_dim) * data.M
    M1 = M_mat @ np.ones(data.l_dim) - M_mat @ alpha
    M2 = -M_mat @ np.ones(data.l_dim) + M_mat @ alpha
    boptimistic.addConstr(slack_expr <= M1)
    boptimistic.addConstr(slack_expr >= M2)
    boptimistic.update()
    boptimistic.write("boptimistic.lp")
    boptimistic.optimize()
    print_solution(boptimistic, data, x, y)

def solve_beck_pessimistic(model_data):
    # Constructed assuming x, y have non-negativity constraints.
    # Initialize Model
    bpessimistic=Model()
    x, y = construct_base_model(bpessimistic, data)
    # Add variables lambda, alpha
    lam = bpessimistic.addMVar(data.l_dim, name="lambda")
    alpha = bpessimistic.addMVar(data.l_dim, vtype=GRB.BINARY, name="alpha")
    bpessimistic.addConstr(data.C @ x + data.D @ y >= data.b)
    bpessimistic.addConstr(data.D.transpose() @ lam <= data.g)
    # complimentary slackness - lam[i] == 0 if alpha[i] == 0 added individually
    for i in range(data.l_dim):
        bpessimistic.addConstr(lam[i] <= data.M * alpha[i])
        bpessimistic.addConstr(lam[i] >= -data.M * alpha[i])
    slack_expr = data.C @ x + data.D @ y - data.b
    M_mat = np.eye(data.l_dim) * data.M
    M1 = M_mat @ np.ones(data.l_dim) - M_mat @ alpha
    M2 = -M_mat @ np.ones(data.l_dim) + M_mat @ alpha
    bpessimistic.addConstr(slack_expr <= M1)
    bpessimistic.addConstr(slack_expr >= M2)
    bpessimistic.update()
    bpessimistic.write("bpessimistic.lp")
    bpessimistic.optimize()
    print_solution(bpessimistic, data, x, y)
    pass

def solve_wrabetz_pessimistic(model_data):
    # Constructed assuming x, y have non-negativity constraints.
    # Initialize Model
    wpessimistic=Model()
    x, y = construct_base_model(wpessimistic, data)
    # Add variables lambda, alpha
    lam = wpessimistic.addMVar(data.l_dim, name="lambda")
    alpha = wpessimistic.addMVar(data.l_dim, vtype=GRB.BINARY, name="alpha")
    wpessimistic.addConstr(data.C @ x + data.D @ y >= data.b)
    wpessimistic.addConstr(data.D.transpose() @ lam <= data.g)
    # complimentary slackness - lam[i] == 0 if alpha[i] == 0 added individually
    for i in range(data.l_dim):
        wpessimistic.addConstr(lam[i] <= data.M * alpha[i])
        wpessimistic.addConstr(lam[i] >= -data.M * alpha[i])
    slack_expr = data.C @ x + data.D @ y - data.b
    M_mat = np.eye(data.l_dim) * data.M
    M1 = M_mat @ np.ones(data.l_dim) - M_mat @ alpha
    M2 = -M_mat @ np.ones(data.l_dim) + M_mat @ alpha
    wpessimistic.addConstr(slack_expr <= M1)
    wpessimistic.addConstr(slack_expr >= M2)
    wpessimistic.update()
    wpessimistic.write("wpessimistic.lp")
    wpessimistic.optimize()
    print_solution(wpessimistic, data, x, y)
    pass

def main():
    example = ModelData()
    beck = return_beck_example()
    solve_nominal(beck)
    beck.log_data()
    P = []
    #solve_beck_optimistic(example, P)

if __name__=="__main__":
    main()
