import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
import sys
import argparse

# returns system of ODEs
def makeODE(l,eCharge,e,h,m,z,u):
    e = e[0]
    # start solving slightly off of z = 0 to avoid 1/0 error
    eps = 1E-8
    dudr = np.zeros(4)
    # define quantities which show up a lot
    hz = 4*h*z/(z**2 + 4)
    hzSq = h*(z**2 - 4)/(z**2 + 4)
    eTerm = eCharge*z/2
    dudr[0] = -1*(eTerm - (l-1)/(z+eps))*u[0] - hz*u[3] - (hzSq - m - e)*u[2]
    dudr[1] = -1*(eTerm - l/(z+eps))*u[1] - hz*u[2] + (hzSq + m + e)*u[3]
    dudr[2] = -1*(-eTerm + l/(z+eps))*u[2] - hz*u[1] - (hzSq - m - e)*u[0]
    dudr[3] = -1*(-eTerm + (l+1)/(z+eps))*u[3] - hz*u[0] + (hzSq + m - e)*u[1]
    return dudr


def shoot(vars, inCond, lParam, eCharge, hParam, mParam, nPts, RMax):
    # Define the mesh
    eParam = vars
    #u3_0, eParam = vars
    z = np.linspace(0., RMax, nPts)

    # Initial conditions

    # Wrapper function for ODEs
    ODEWrap = lambda z, u: makeODE(lParam, eCharge, eParam, hParam, mParam, z, u)

    # Solve the ODEs
    solution = solve_ivp(ODEWrap, [0., RMax], inCond, method='RK45', t_eval=z, rtol=1e-8, atol=1e-10)

    # Check if the solution was successful
    if solution.success:
        # Extract the values at z = RMax
        # For root finding, can only return as many residuals as parameters to fit
        #finalVals = np.abs(solution.y[0,-1])+np.abs(solution.y[1,-1])+np.abs(solution.y[2,-1])+np.abs(solution.y[3,-1])
        finalVals = solution.y[3,-1]
        return finalVals
    else:
        # Return none if solution not found
        print("Solution not found.")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("l", help="angular momenta, l = 1, -1 allowed",type=int)
    parser.add_argument("e", help="electric charge of fermion", type=float)
    parser.add_argument("E", help="initial guess for energy", type=float)
    parser.add_argument("h", help="coupling strength", type=float)
    parser.add_argument("m", help="mass of fermion", type=float)
    parser.add_argument("-NPts", help="number of points for mesh", default=1000, type=int)
    parser.add_argument("-RMax", help="RMax", default=10, type=float)
    args = parser.parse_args()

    if (args.l == 1):
        inCond = [0.,0.,0.,1.]
        shootWrap = lambda vars : shoot(vars, inCond, args.l, args.e, args.h, args.m, args.NPts, args.RMax)
        sol = root(shootWrap, args.E, method='hybr', tol=1e-10)
        eSol = sol.x
        print(eSol)

    elif (args.l == -1):
        inCond = [1.,0.,0.,0.]
        shootWrap = lambda vars : shoot(vars, inCond, args.l, args.e, args.h, args.m, args.NPts, args.RMax)
        sol = root(shootWrap, args.E, method='hybr', tol=1e-10)
        eSol = sol.x
        print(eSol[0])
    
    else:
        print("l must be 1 or -1, exiting")

    ODEWrap = lambda z, u: makeODE(args.l, args.e, eSol, args.h, args.m, z, u)
    z = np.linspace(0.,args.RMax,args.NPts)
    solFinal = solve_ivp(ODEWrap, [0., args.RMax], inCond, method='RK45', t_eval=z, rtol=1e-8, atol=1e-10)
    u_sol = solFinal.y
    plt.figure(figsize=(10, 6))
    plt.plot(z, u_sol[0], label='u1(z)')
    plt.plot(z, u_sol[1], label='u2(z)')
    plt.plot(z, u_sol[2], label='u3(z)')
    plt.plot(z, u_sol[3], label='u4(z)')
    plt.xlabel('z')
    plt.ylabel('u')
    plt.title(f"E: {eSol[0]}, l: {args.l}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return 0

if __name__ == "__main__":
    main()