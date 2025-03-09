from flowinn.training.steady_loss import SteadyNavierStokesLoss

class NavierStokesLoss:
    def __init__(self, loss_type = 'steady', mesh=None, model=None, **kwargs):
        """For backwards compatibility, redirects to create method"""
        if mesh is not None and model is not None:
            return self.create(loss_type, mesh, model, **kwargs)
        return None

    @staticmethod
    def create(loss_type: str, mesh, model, **kwargs):
        """Factory method to create appropriate loss function"""
        if loss_type.lower() == 'steady':
            return SteadyNavierStokesLoss(mesh, model, **kwargs)
        elif loss_type.lower() == 'unsteady':
            raise NotImplementedError("Unsteady Navier-Stokes loss not yet implemented")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def __new__(cls, mesh=None, model=None, **kwargs):
        """Override new to return the actual loss object"""
        if mesh is not None and model is not None:
            return cls.create('steady', mesh, model, **kwargs)
        return super().__new__(cls)

