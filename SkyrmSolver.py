import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
import argparse

# returns system of ODEs
def makeODE(l,eCharge,e,h,m,z,u):
    dudr = np.zeros(4)
    # define quantities which show up a lot
    hz = 4*h*z/(z**2 + 4)
    hzSq = h*(z**2 - 4)/(z**2 + 4)
    eTerm = eCharge*z/2
    dudr[0] = -1*(eTerm - (l-1)/z)*u[0] - hz*u[3] - (hzSq - m - e)*u[2]
    dudr[1] = -1*(eTerm - l/z)*u[1] - hz*u[2] + (hzSq + m + e)*u[3]
    dudr[2] = -1*(-eTerm + l/z)*u[2] - hz*u[1] - (hzSq - m - e)*u[0]
    dudr[3] = -1*(-eTerm + (l+1)/z)*u[3] - hz*u[0] + (hzSq + m - e)*u[1]
    return dudr


def shoot(vars, eps, lParam, eCharge, hParam, mParam, nPts, RMax):
    # Define the mesh
    z = np.linspace(eps, RMax, nPts)

    # Unpack vars to shoot over
    initU, eParam = vars

    # set up initial conditons (dependent on value of l)
    if (lParam == 1):
        # u1 ~ c1 r^2, u2 ~ c2 r, u3 ~ c3 r, u4 ~ c4
        # scale so c1 = 1, then have two consistency relations, shoot over c4
        c1 = 1.
        c3 = -1. * c1 * (eParam - hParam - mParam) / 2
        c4 = initU
        c2 = 2 * hParam * c3 / ((eParam - mParam)**2 - hParam**2) - 4 * c4 / (eParam + hParam - mParam)
        inCond = [c1*eps**2,c2*eps,c3*eps,c4]
    
    if (lParam == -1):
        # u1 ~ c1, u2 ~ c2 r, u3 ~ c3 r, u4 ~ c4 r^2
        # scale so c4 = 1, then have two consistency relations, shoot over c1
        c1 = initU
        c4 = 1.
        c2 = c4 * (eParam - hParam + mParam) / 2
        c3 = (c1 + hParam * c4 / 4)* 4 / (eParam + hParam + mParam)
        inCond = [c1,c2*eps,c3*eps,c4*eps**2]
    
    if (lParam == 0):
        # u1 ~ c1 r, u2 ~ c2, u3 ~ c3, u4 ~ c4 r
        # scale so c1 = 1, then have two consistency relations, shoot over c2
        c1 = 1
        c2 = initU
        c3 = c1 * 2 / (eParam + hParam + mParam)
        c4 = c2 * (eParam + hParam - mParam)/(-2)
        inCond = [c1*eps,c2,c3,c4*eps]

    # Wrapper function for ODEs
    ODEWrap = lambda z, u: makeODE(lParam, eCharge, eParam, hParam, mParam, z, u)

    # Solve the ODEs
    solution = solve_ivp(ODEWrap, [eps, RMax], inCond, method='LSODA', t_eval=z, rtol=1e-10, atol=1e-10)

    # Check if the solution was successful
    if solution.success:
        # Extract the values at z = RMax
        # For root finding, can only return as many residuals as parameters to fit
        finalVals = np.array([solution.y[0,-1],solution.y[1,-1]])
        return finalVals
    else:
        # Return none if solution not found
        print("Solution not found.")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("l", help="angular momenta, l = 1, -1 allowed",type=int)
    parser.add_argument("E", help="initial guess for energy", type=float)
    parser.add_argument("u0", help="initial guess for function value", type=float)
    parser.add_argument("e", help="electric charge of fermion", type=float)
    parser.add_argument("h", help="coupling strength", type=float)
    parser.add_argument("m", help="mass of fermion", type=float)
    parser.add_argument("-NPts", help="number of points for mesh", default=1000, type=int)
    parser.add_argument("-RMax", help="RMax", default=10, type=float)
    args = parser.parse_args()

    # don't start solving at 0, avoid 1/0 errors
    eps = 1E-5

    shootWrap = lambda vars : shoot(vars, eps, args.l, args.e, args.h, args.m, args.NPts, args.RMax)
    sol = root(shootWrap, np.array([args.u0,args.E]), method='hybr', tol=1e-15)
    u0Sol, eSol = sol.x
    print("u0 = " + str(u0Sol) + ", E = " + str(eSol))

    if (args.l == 1):
        # u1 ~ c1 r^2, u2 ~ c2 r, u3 ~ c3 r, u4 ~ c4
        c1 = 1.
        c3 = -1. * c1 * (eSol - args.h - args.m) / 2
        c4 = u0Sol
        c2 = 2 * args.h * c3 / ((eSol - args.m)**2 - args.h**2) - 4 * c4 / (eSol + args.h - args.m)
        inCond = [c1*eps**2,c2*eps,c3*eps,c4]
    
    if (args.l == -1):
        # u1 ~ c1, u2 ~ c2 r, u3 ~ c3 r, u4 ~ c4 r^2
        c1 = u0Sol
        c4 = 1.
        c2 = c4 * (eSol - args.h + args.m) / 2
        c3 = (c1 + args.h * c4 / 4)* 4 / (eSol + args.h + args.m)
        inCond = [c1,c2*eps,c3*eps,c4*eps**2]
    
    if (args.l == 0):
        # u1 ~ c1 r, u2 ~ c2, u3 ~ c3, u4 ~ c4 r
        c1 = 1
        c2 = u0Sol
        c3 = c1 * 2 / (eSol + args.h + args.m)
        c4 = c2 * (eSol + args.h - args.m)/(-2)
        inCond = [c1*eps,c2,c3,c4*eps]

    ODEWrap = lambda z, u: makeODE(args.l, args.e, eSol, args.h, args.m, z, u)
    z = np.linspace(eps,args.RMax,args.NPts)
    solFinal = solve_ivp(ODEWrap, [eps, args.RMax], inCond, method='LSODA', t_eval=z, rtol=1e-10, atol=1e-10)
    u_sol = solFinal.y
    plt.figure(figsize=(10, 6))
    plt.plot(z, u_sol[0], label='u1(z)')
    plt.plot(z, u_sol[1], label='u2(z)')
    plt.plot(z, u_sol[2], label='u3(z)')
    plt.plot(z, u_sol[3], label='u4(z)')
    plt.xlabel('z')
    plt.ylabel('u')
    plt.title(f"E: {eSol:.3f}, l: {args.l}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return 0

if __name__ == "__main__":
    main()