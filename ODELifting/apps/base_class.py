class BVP:

    def __init__(self):
        pass

    def get_ode(self):
        raise NotImplementedError("ODE is not defined!")

    def get_boundary_fct(self):
        raise NotImplementedError("Boundary function is not defined!")

    def get_init(self):
        raise NotImplementedError("No initial values given!")

    def get_grid_details(self):
        raise NotImplementedError("No time grid details given!")

