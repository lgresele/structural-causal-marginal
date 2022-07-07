import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

def generate_instance(beta = False, parameters_beta = np.array([1.0, 1.0])):
    # Generate a random instance of joint distribution P_{XYZ} for the structural causal marginal problem
    
    # Parameters of the marginal distributions of X and Y
    theta_x = np.random.uniform(low=0.0, high=1.0) #P(X=1)
    theta_y = np.random.uniform(low=0.0, high=1.0) #P(Y=1)

    # Parameters of the conditional distribution of Z given X and Y
    if beta == False:
        theta_z_00 = np.random.uniform(low=0.0, high=1.0) #P(Z=1 | X=0, Y=0)
        theta_z_01 = np.random.uniform(low=0.0, high=1.0) #P(Z=1 | X=0, Y=1)
        theta_z_10 = np.random.uniform(low=0.0, high=1.0) #P(Z=1 | X=1, Y=0)
        theta_z_11 = np.random.uniform(low=0.0, high=1.0) #P(Z=1 | X=1, Y=1)
    else:
        theta_z_00 = np.random.beta(a= parameters_beta[0], b= parameters_beta[1]) #P(Z=1 | X=0, Y=0)
        theta_z_01 = np.random.beta(a= parameters_beta[0], b= parameters_beta[1]) #P(Z=1 | X=0, Y=1)
        theta_z_10 = np.random.beta(a= parameters_beta[0], b= parameters_beta[1]) #P(Z=1 | X=1, Y=0)
        theta_z_11 = np.random.beta(a= parameters_beta[0], b= parameters_beta[1]) #P(Z=1 | X=1, Y=1)
    
    return theta_x, theta_y, theta_z_00, theta_z_01, theta_z_10, theta_z_11

def generate_instance_and_model(theta_x, theta_y):
    # Given the marginal distributions of X and Y, 
    # Assuming a AND model on Y -> Z, 
    # Assuming X independent of Z,
    # This function outputs:
    # Parameters of the conditional distribution of Z given X and Y
    
    theta_z_00 = 0.0 #P(Z=1 | X=0, Y=0)
    theta_z_01 = .5/theta_y #P(Z=1 | X=0, Y=1)
    theta_z_10 = 0.0 #P(Z=1 | X=1, Y=0)
    theta_z_11 = .5/theta_y  #P(Z=1 | X=1, Y=1)

    # Compute the marginal probability of Z
    theta_z = (1-theta_y)*(1-theta_x)*theta_z_00 + (1-theta_y)*(theta_x)*theta_z_10 + theta_y*(1-theta_x)*theta_z_01 + theta_y*(theta_x)*theta_z_11
    return theta_x, theta_y, theta_z_00, theta_z_01, theta_z_10, theta_z_11, theta_z

def get_alpha(theta_x, theta_y, theta_z_00, theta_z_01, theta_z_10, theta_z_11):
    '''
    Starts from these parameters:
    - theta_x = P(X=1)
    - theta_y = P(Y=1)
    - theta_z_00 = P(Z=1 | X=0, Y=0)
    - theta_z_01 = P(Z=1 | X=0, Y=1)
    - theta_z_10 = P(Z=1 | X=1, Y=0)
    - theta_z_11 = P(Z=1 | X=1, Y=1)
    And returns these parameters:
    - alpha_xz_00 = P(X=0, Z=0)
    - alpha_xz_01 = P(X=0, Z=1)
    - alpha_xz_10 = P(X=1, Z=0)
    - alpha_xz_11 = P(X=1, Z=1)
    '''
    #P(X=0, Z=0) = P(X=0)[P(Y=0)P(Z=0|X=0, Y = 0) + P(Y=1)P(Z=0|X=0, Y = 1)]
    alpha_xz_00 = (1.0 -theta_x)*( (1.0 -theta_y)*(1.0 - theta_z_00) + theta_y * (1.0 - theta_z_01) )
    #P(X=0, Z=1) = P(X=0)[P(Y=0)P(Z=1|X=0, Y = 0) + P(Y=1)P(Z=1|X=0, Y = 1)]
    alpha_xz_01 = (1.0 -theta_x)*( (1.0 -theta_y)*theta_z_00 + theta_y * theta_z_01 )
    #P(X=1, Z=0) = P(X=1)[P(Y=0)P(Z=0|X=1, Y = 0) + P(Y=1)P(Z=0|X=1, Y = 1)]
    alpha_xz_10 = theta_x*((1.0 -theta_y)* (1- theta_z_10) + theta_y* (1- theta_z_11))
    #P(X=1, Z=1) = P(X=1)[P(Y=0)P(Z=1|X=1, Y = 0) + P(Y=1)P(Z=1|X=1, Y = 1)]                             
    alpha_xz_11 = theta_x*((1.0 -theta_y)* theta_z_10 + theta_y* theta_z_11)

    alpha = np.array([alpha_xz_00, alpha_xz_01, alpha_xz_10, alpha_xz_11 ] )
    return alpha

def get_beta(theta_x, theta_y, theta_z_00, theta_z_01, theta_z_10, theta_z_11):
    '''
    Starts from these parameters:
    - theta_x = P(X=1)
    - theta_y = P(Y=1)
    - theta_z_00 = P(Z=1 | X=0, Y=0)
    - theta_z_01 = P(Z=1 | X=0, Y=1)
    - theta_z_10 = P(Z=1 | X=1, Y=0)
    - theta_z_11 = P(Z=1 | X=1, Y=1)
    And returns these parameters:
    - beta_yz_00 = P(Y=0, Z=0)
    - beta_yz_01 = P(Y=0, Z=1)
    - beta_yz_10 = P(Y=1, Z=0)
    - beta_yz_11 = P(Y=1, Z=1)
    '''
    #P(Y=0, Z=0) = P(Y=0)[P(X=0)P(Z=0|X=0, Y = 0) + P(X=1)P(Z=0|X=1, Y = 0)]
    beta_yz_00 = (1.0 - theta_y)* ( (1.0 - theta_x)* (1.0 - theta_z_00 ) + theta_x* (1.0 - theta_z_10 ) )
    #P(Y=0, Z=1) = P(Y=0)[P(X=0)P(Z=1|X=0, Y = 0) + P(X=1)P(Z=1|X=1, Y = 0)]
    beta_yz_01 = (1.0 - theta_y)* ( (1.0 - theta_x)*theta_z_00  + theta_x*theta_z_10  )
    #P(Y=1, Z=0) = P(Y=1)[P(X=0)P(Z=0|X=0, Y = 1) + P(X=1)P(Z=0|X=1, Y = 1)]
    beta_yz_10 = theta_y * ( (1.0 - theta_x)* (1.0 - theta_z_01 ) + theta_x* (1.0 - theta_z_11 ) )
    #P(Y=1, Z=1) = P(Y=1)[P(X=0)P(Z=1|X=0, Y = 1) + P(X=1)P(Z=1|X=1, Y = 1)]
    beta_yz_11 = theta_y * ( (1.0 - theta_x)*theta_z_01  + theta_x*theta_z_11  )

    beta = np.array([beta_yz_00, beta_yz_01, beta_yz_10, beta_yz_11 ] )
    return beta

def get_trivial_bounds(gamma):
    # Returns parameter of the canonical representation given the vectors alpha or beta
    c_0 = np.array([0.0, 
                1.0 - gamma[0]/(gamma[0] + gamma[1]) - gamma[2]/(gamma[2] + gamma[3]),
                gamma[0]/(gamma[0] + gamma[1]),
                gamma[2]/(gamma[2] + gamma[3])
               ])
    
    LB_C = np.max([0.0, -c_0[1] ])
    UB_C = np.min([c_0[2], c_0[3] ])
    
    return LB_C, UB_C, c_0

def get_constraint_matrices(theta_x, theta_y):
    # Builds the constraint matrices for our linear program
    A = np.array([ [1.0, theta_y, 1.0 - theta_y, 0.0, theta_y, theta_y, 0.0, 0.0, (1.0 - theta_y), 0.0, (1.0 - theta_y),0.0, 0.0, 0.0, 0.0, 0.0 ],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - theta_y, 0.0, 1.0 - theta_y, 0.0, 0.0, theta_y, theta_y, 0.0, 1.0 - theta_y, theta_y, 1.0],
             [0.0, 0.0, 0.0, 0.0, 1.0 - theta_y, 0.0, 1.0 - theta_y, 0.0, theta_y, theta_y, 0.0, 0.0, 1.0, theta_y, 1.0 - theta_y, 0.0],
             [0.0, 1.0-theta_y, theta_y, 1.0, 0.0, 0.0, theta_y, theta_y, 0.0, 1.0-theta_y, 0.0, 1.0-theta_y, 0.0 , 0.0, 0.0, 0.0]
            ])# These entries depend on theta_y
# 4 rows, 16 columns

    B = np.array([ [1.0, theta_x, theta_x, theta_x, 1.0 - theta_x,  0.0, 0.0, 0.0, 1.0- theta_x, 0.0, 0.0, 0.0, 1.0- theta_x, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0 - theta_x, 0.0, 0.0, 0.0, 1.0- theta_x, 0.0, 0.0, 0.0, 1.0-theta_x, theta_x, theta_x, theta_x, 1.0],
             [0.0, 0.0, 1.0- theta_x, 0.0, 0.0, 0.0, 1.0- theta_x, 0.0, theta_x, theta_x, 1.0, theta_x, 0.0, 0.0, 1.0-theta_x, 0.0],
             [0.0, 1.0-theta_x, 0.0, 0.0, theta_x, 1.0, theta_x, theta_x, 0.0, 1.0-theta_x, 0.0, 0.0, 0.0, 1.0 - theta_x, 0.0, 0.0]
            ])# These entries depend on theta_x
# 4 rows, 16 columns

    return A, B

def get_linear_prog_bounds(a_0, b_0, A, B, lambda_x_min, lambda_x_max, lambda_y_min, lambda_y_max):
    constr = np.array([0.0]*16 + [1.0] + [0.0])
    A_eq = np.vstack([np.vstack([A, B]), np.array([1.0]*16) ])
    A_eq = np.c_[A_eq, np.array([-1.0]*2 + [1.0]*2 + [0.0]*5).T]
    A_eq = np.c_[A_eq, np.array([0.0]*4 + [-1.0]*2 + [1.0]*2 + [0.0]).T]
    
    b_eq = np.concatenate((a_0, b_0))
    b_eq = np.concatenate((b_eq, np.array([1.0])))
    
    bounds = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (lambda_x_min, lambda_x_max),
        (lambda_y_min, lambda_y_max)
        ]
    
    lambda_min_A = linprog(constr.T, 
              A_eq=A_eq, 
              b_eq=b_eq,
              bounds=bounds, 
              method='revised simplex',
                          )
    
    lambda_max_A = linprog(-constr.T, 
              A_eq=A_eq, 
              b_eq=b_eq,
              bounds=bounds, 
              method='revised simplex',
                          )
    
    constr_B = np.array([0.0]*17 + [1.0])
    
    lambda_min_B = linprog(constr_B.T, 
              A_eq=A_eq, 
              b_eq=b_eq,
              bounds=bounds, 
              method='revised simplex',
                          )
    
    lambda_max_B = linprog(-constr_B.T, 
              A_eq=A_eq, 
              b_eq=b_eq,
              bounds=bounds, 
              method='revised simplex',
                          )
    
    return lambda_min_A['x'][16], lambda_max_A['x'][16], lambda_min_B['x'][17], lambda_max_B['x'][17]

def compute_inequality_params(A, B, lambda_y_min, lambda_y_max, b_0, lambda_x_min, lambda_x_max, a_0):
    '''
    Computes the parameters of the inequality defining points in the high-dimensional polytope of joint SCMs.
    
    Input:
    A: constraint matrix X->Z
    B: constraint matrix Y->Z 
    lambda_y_min: "Trivial" lower bound for the marginal Y->Z SCM (only marginal information)
    lambda_y_max: "Trivial" upper bound for the marginal Y->Z SCM (only marginal information) 
    b_0: Constant offset vector for Y->Z marginal models
    lambda_x_min: "Trivial" lower bound for the marginal X->Z SCM (only marginal information) 
    lambda_x_max: "Trivial" upper bound for the marginal X->Z SCM (only marginal information) 
    a_0:  Constant offset vector for X->Z marginal models
    
    Output:
    A_ineq: a matrix
    b_ineq: a constant offset
    
    '''
    A_ineq = np.vstack([A, -A, B, -B, -np.eye(16)])

    first_bit = a_0 + np.array([ lambda_x_max, lambda_x_max, -lambda_x_min, -lambda_x_min])
    second_bit = -a_0 - np.array([ lambda_x_min, lambda_x_min, -lambda_x_max, -lambda_x_max])
    third_bit = b_0 + np.array([ lambda_y_max, lambda_y_max, -lambda_y_min, -lambda_y_min])
    fourth_bit = -b_0 - np.array([ lambda_y_min, lambda_y_min, -lambda_y_max, -lambda_y_max])
    b_ineq = np.hstack([first_bit, second_bit, third_bit, fourth_bit, np.zeros(16)])
    return A_ineq, b_ineq

def compute_equality_params(A, B, a_0, b_0):
    '''
    Computes the parameters of the equality
    
    For our problem, they correspond to the normalisation constraints;
    additionally, we have to consider how the constraints a(\lambda^A)=Ac need to be enforced (and the same for B)
    
    Input:
    A: constraint matrix X->Z
    B: constraint matrix Y->Z 
    a_0:  Constant offset vector for X->Z marginal models
    b_0: Constant offset vector for Y->Z marginal models

    Output:
    C: a matrix
    d: a constant vector 
    '''
    const_matrix = np.array([[1.0, -1.0, 0.0, 0.0],
                            [1.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 1.0]])    

    # Equality constraints A, a_0
    C_A = const_matrix @ A
    d_A = const_matrix @ a_0
    
    # Equality constraints B, b_0
    C_B = const_matrix @ B
    d_B = const_matrix @ b_0
    
    C_norm = np.array([[1.0]*16])
    d_norm = np.array([1.0])
    
    C = np.vstack([C_A, C_B, C_norm])
    d = np.hstack([d_A, d_B, d_norm])
    
    return (C, d)

def compute_affine_projection(A, B, a_0, b_0):
    '''
    Computes the parameters of the affine projection
    
    Input:
    A: constraint matrix X->Z
    B: constraint matrix Y->Z 
    a_0:  Constant offset vector for X->Z marginal models
    b_0: Constant offset vector for Y->Z marginal models

    Output:
    E: a projection matrix
    f: a constant offset for the projection
    
    '''
    two_d_proj = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
                        )
    # E is equal to two_d_proj @ stacked(A, B) matrices
    E = two_d_proj @ np.vstack([A, B])
    # f is equal to two_d_proj @ stacked(a_0, b_0)
    f = -two_d_proj @ np.hstack([a_0, b_0])
    return E, f

# Functions to compute the area of the projected polytope given its vertices:

def explode_xy(xy):
    xl=[]
    yl=[]
    for i in range(len(xy)):
        xl.append(xy[i][0])
        yl.append(xy[i][1])
    return xl,yl    
    
def shoelace_area(x_list,y_list):
    a1,a2=0,0
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    for j in range(len(x_list)-1):
        a1 += x_list[j]*y_list[j+1]
        a2 += y_list[j]*x_list[j+1]
    l=abs(a1-a2)/2
    return l

def area_polygon_points(points):
    hull = ConvexHull(points)
    xy_e = explode_xy(points[hull.vertices])
    area_polygon=shoelace_area(xy_e[0],xy_e[1])
    return area_polygon