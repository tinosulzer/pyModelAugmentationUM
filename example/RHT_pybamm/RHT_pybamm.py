import casadi
import numpy as np
from os import path
import pybamm
import os
import matplotlib.pyplot as plt
from pybamm_model import RHTModel


class RHT:

    # Class to generate data from the Radiative heat transfer model

    def __init__(
        self,
        T_inf=5.0,
        npoints=129,
        tol=1e-8,
        lambda_reg=1e-5,
        plot=True,
        savesol=False,
    ):

        self.T_inf = T_inf  # Temperature of the one-dimensional body
        # Boolean flag whether to plot the solution at the end of simulation
        self.plot = plot
        # Boolean flag whether to save the converged temperatures
        self.savesol = savesol
        self.lambda_reg = lambda_reg  # Regularization constant for objective function
        # Initialize beta
        self.beta = 1
        # Initialize flags
        self.has_obj_and_jac_funs = False

        # Define model
        model = RHTModel()

        # Define settings
        parameter_values = model.default_parameter_values
        parameter_values.update({"T_inf": T_inf, "beta": "[input]"})
        var_pts = {model.x: npoints}
        solver = pybamm.CasadiAlgebraicSolver(tol=tol)

        # Create simulation
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, solver=solver, var_pts=var_pts
        )
        t_eval = [0]

        sim.solve(t_eval, inputs={"beta": "[sym]{}".format(npoints)})
        self.sol = sim.solution
        self.T = self.sol["Temperature"]
        self.x = self.T.x_sol

        fig, self.ax = plt.subplots()

    # ----------------------------------------------------------------------------------

    def direct_solve(self):

        T = self.T.value({"beta": self.beta}).full()

        if self.savesol is True:
            save_folder = "example/RHT_pybamm/Model_solutions"
            os.makedirs(save_folder, exist_ok=True)
            print("Saving solution to file")
            np.savetxt("{}/solution_{}".format(save_folder, self.T_inf), T)

        # Once the simulation is terminated, show the results if plot is True

        if self.plot is True:
            ax = self.ax
            ax.plot(self.x, T)
            ax.set_xlabel("x")
            ax.set_ylabel("Temperature")
            plt.show()

        return T

    # ----------------------------------------------------------------------------------

    def adjoint_solve(self, data):
        if not self.has_obj_and_jac_funs:
            self.create_obj_and_jac_funs(data)
        
        return self.jac_fun(self.beta).full().flatten()

    # ----------------------------------------------------------------------------------

    def getObj(self, data):
        if not self.has_obj_and_jac_funs:
            self.create_obj_and_jac_funs(data)

        return self.obj_fun(self.beta).full()

    def create_obj_and_jac_funs(self, data):
        # Create objective function and derivative
        beta = self.sol.inputs["beta"]
        T = self.T.value({"beta": beta})
        obj = casadi.sum1((T - data) ** 2) + self.lambda_reg * casadi.sum1((beta - 1.0) ** 2)
        self.obj_fun = casadi.Function("obj", [beta], [obj])

        jac = casadi.jacobian(obj, beta)
        self.jac_fun = casadi.Function("jacfn", [beta], [jac])

        self.has_obj_and_jac_funs = True

if __name__ == "__main__":
    from scipy.optimize import minimize

    # for T_inf in np.linspace(5.0, 50.0, 10):
    #     rht = RHT(T_inf=T_inf, savesol=True, plot=False)
    #     rht.direct_solve()

    # Try using scipy.minimize
    rht = RHT(T_inf=10, savesol=False, plot=False, lambda_reg=10)
    rht.direct_solve()
    data = rht.direct_solve()

    def objective(x):
        rht.beta = x
        return rht.getObj(data)

    def jac(x):
        rht.beta = x
        return rht.adjoint_solve(data)

    timer = pybamm.Timer()
    sol = minimize(objective, [1.1]*129)
    print("Without jac: ", timer.time())
    timer.reset()
    sol = minimize(objective, [1.1]*129, jac=jac)
    print("With jac: ", timer.time())
    # print(sol)
