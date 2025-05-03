import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
import sys

# returns system of ODEs
def makeODE(l,eCharge,e,h,m,z,u):
    eps = 0.00001
    dudr = np.zeros(2)
    # define quantities which show up a lot
    hz = 4*h*z/(z**2 + 4)
    hzSq = h*(z**2 - 4)/(z**2 + 4)
    eTerm = eCharge*z/2
    '''
    dudr[0] = -1*(eTerm - (l-1)/(z+eps))*u[0] - hz*u[3] - (hzSq - m - e)*u[2]
    dudr[1] = -1*(eTerm - l/(z+eps))*u[1] - hz*u[2] + (hzSq + m + e)*u[3]
    dudr[2] = (eTerm - l/(z+eps))*u[2] - hz*u[1] - (hzSq - m + e)*u[0]
    dudr[3] = (eTerm - (l+1)/(z+eps))*u[3] - hz*u[0] + (hzSq + m - e)*u[1]
    '''
    dudr[0] = -1*(eTerm - (l-1)/(z+eps))*u[0] + hz*u[0] - (hzSq - m - e)*u[1]
    dudr[1] = -1*(eTerm - l/(z+eps))*u[1] - hz*u[1] - (hzSq + m + e)*u[0]

    return dudr


def shoot(vars, lParam, eCharge, hParam, mParam, nPts, RMax):
    # Define the mesh
    eParam = vars
    #u3_0, eParam = vars
    z = np.linspace(0., RMax, nPts)

    # Initial conditions
    #initialCond = [0, 1, u3_0, 0]  # Assuming u1(0) = 0, u2(0) = 1, u3(0) = u3_0, u4(0) = 0
    initialCond = [0, 1]

    # Wrapper function for ODEs
    ODEWrap = lambda z, u: makeODE(lParam, eCharge, eParam, hParam, mParam, z, u)

    # Solve the ODEs
    solution = solve_ivp(ODEWrap, [0., RMax], initialCond, method='RK45', t_eval=z, rtol=1e-8, atol=1e-10)

    # Check if the solution was successful
    if solution.success:
        # Extract the values at z = RMax
        # For root finding, can only return as many residuals as parameters to fit
        # Can only cause two to go to zero, other solutions should follow
        finalVals = np.abs(solution.y[0,-1])+np.abs(solution.y[1,-1])
        return finalVals
    else:
        # Return none if solution not found
        print("Solution not found.")
        return None

def main():
    # read in command line args
    # order is l, e charge, e, u3_0, h, m, nPts
    # nPts is number of points on mesh
    lParam = float(sys.argv[1])
    eParam = float(sys.argv[2])
    eCharge = float(sys.argv[3])
    u3_0 = float(sys.argv[4])
    hParam = float(sys.argv[5])
    mParam = float(sys.argv[6])
    nPts = int(sys.argv[7])

    # Boundary at inf
    RMax = 20.

    # right now, just doing l = 0
    # this sets u1 and u4 to 0 at origin
    # can scale such that u2 = 1 at origin
    # then u3 value at origin is free param, as is energy

    # wrapper for shooting
    shootWrap = lambda vars : shoot(vars, lParam, eCharge, hParam, mParam, nPts, RMax)
    sol = root(shootWrap,eParam,method='hybr',tol=1e-10)
    eSol = sol.x
    print(eSol)

    ODEWrap = lambda z, u: makeODE(lParam, eCharge, eSol, hParam, mParam, z, u)
    #initialCond = [0,1,u30Sol,0]
    initialCond = [0,1]
    z = np.linspace(0.,RMax,nPts)
    solFinal = solve_ivp(ODEWrap, [0., RMax], initialCond, method='RK45', t_eval=z, rtol=1e-8, atol=1e-10)
    
    u_sol = solFinal.y

    plt.figure(figsize=(10, 6))
    plt.plot(z, u_sol[0], label='u1(z)')
    plt.plot(z, u_sol[1], label='u2(z)')
    #plt.plot(z, u_sol[2], label='u3(z)')
    #plt.plot(z, u_sol[3], label='u4(z)')
    plt.xlabel('z')
    plt.ylabel('u')
    plt.title('Solutions of the ODE System')
    plt.legend()
    plt.grid(True)
    plt.show()

    return 0

if __name__ == "__main__":
    main()