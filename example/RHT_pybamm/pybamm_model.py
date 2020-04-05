#
# PyBaMM implementation of the RHT model
#
import pybamm
import numpy as np

class RHTModel(pybamm.BaseModel):
    def __init__(self, options=None):
        super().__init__()

        self.name = "RHT"
        T = pybamm.Variable("Temperature", domain="line")
        self.x = pybamm.SpatialVariable("x", domain="line")

        # Define parameters beta
        T_inf = pybamm.FunctionParameter("T_inf", {"x": self.x})
        # beta = pybamm.Parameter("beta")
        eps0 = pybamm.Parameter("eps0")

        beta = pybamm.FunctionParameter("beta", {"Temperature": T})
        # Also make a beta_decoupled, for plotting
        beta_decoupled = pybamm.FunctionParameter("beta_decoupled", {"Temperature": T})

        # Define model
        dTdx = pybamm.grad(T)
        source = beta * eps0 * (T_inf ** 4 - T ** 4)

        self.algebraic = {T: pybamm.div(dTdx) + source}
        # Careful initialization as solution is not unique for large T_inf
        self.initial_conditions = {T: 0.7 * T_inf}
        self.boundary_conditions = {
            T: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        self.variables = {
            "Temperature": T,
            "beta": beta,
            "x": self.x,
            "x [m]": self.x,
            "beta_decoupled": beta_decoupled,
        }

    @property
    def default_parameter_values(self):
        # Define default parameter values with true beta
        def beta(T):
            T_inf = pybamm.FunctionParameter("T_inf", {"x": self.x})
            h = pybamm.Parameter("h")
            eps0 = pybamm.Parameter("eps0")
            return (
                1e-4
                * (1.0 + 5.0 * pybamm.sin(3 * np.pi * T / 200.0) + pybamm.exp(0.02 * T))
                / eps0
                + h * (T_inf - T) / (T_inf ** 4 - T ** 4) / eps0
            )

        return pybamm.ParameterValues(
            {
                "C-rate": 0,
                "Cell capacity [A.h]": 1,
                "T_inf": 30,
                "beta": beta,
                "beta_decoupled": beta,
                "eps0": 5e-4,
                "h": 0.5,
            }
        )

    @property
    def default_geometry(self):
        geometry = {
            "line": {
                "primary": {self.x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            },
        }
        return geometry

    @property
    def default_var_pts(self):
        var_pts = {self.x: 50}
        return var_pts

    @property
    def default_submesh_types(self):
        submesh_types = {"line": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}
        return submesh_types

    @property
    def default_spatial_methods(self):
        spatial_methods = {"line": pybamm.FiniteVolume()}
        return spatial_methods

    @property
    def default_solver(self):
        return pybamm.CasadiAlgebraicSolver()


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")

    true_model = RHTModel()
    true_model.name = "True RHT"
    parameter_values = true_model.default_parameter_values
    parameter_values["T_inf"] = 45
    sim_true = pybamm.Simulation(true_model, parameter_values=parameter_values)

    base_model = RHTModel()
    base_model.name = "Base RHT"
    parameter_values = base_model.default_parameter_values
    parameter_values["T_inf"] = 45
    parameter_values["beta"] = 1
    sim_base = pybamm.Simulation(base_model, parameter_values=parameter_values)

    t_eval = np.array([0, 1])
    sims = []
    for sim in [sim_true, sim_base]:
        sim.solve(t_eval)
        sims.append(sim)

    pybamm.dynamic_plot(
        sims, ["Temperature", "beta", "beta_decoupled"], spatial_unit="m"
    )
