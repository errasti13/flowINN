from flowinn.training.steady_loss import SteadyNavierStokesLoss
from flowinn.training.unsteady_loss import UnsteadyNavierStokesLoss

class NavierStokesLoss:
    def __init__(self, loss_type = 'steady', mesh=None, model=None, **kwargs):
        """For backwards compatibility, redirects to create method"""
        if mesh is not None and model is not None:
            return self.create(loss_type, mesh, model, **kwargs)
        return None

    @staticmethod
    def create(loss_type: str, mesh, model, **kwargs):
        """
        Factory method to create appropriate loss function.
        
        Returns:
            A callable loss function object
        """
        if loss_type.lower() == 'steady':
            loss_obj = SteadyNavierStokesLoss(mesh, model, **kwargs)
        elif loss_type.lower() == 'unsteady':
            loss_obj = UnsteadyNavierStokesLoss(mesh, model, **kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        # Ensure it's callable
        if not callable(loss_obj):
            # If the loss object isn't callable, wrap it in a callable function
            original_loss_obj = loss_obj
            
            class CallableLoss:
                def __init__(self, loss_obj):
                    self.loss_obj = loss_obj
                    
                def __call__(self, batch_data=None):
                    if hasattr(self.loss_obj, 'loss_function'):
                        return self.loss_obj.loss_function(batch_data)
                    else:
                        raise AttributeError("Loss object has no loss_function method")
                        
            loss_obj = CallableLoss(original_loss_obj)
            
        return loss_obj

    def __new__(cls, loss_type='steady', mesh=None, model=None, **kwargs):
        """Override new to return the actual loss object"""
        if mesh is not None and model is not None:
            return cls.create(loss_type, mesh, model, **kwargs)
        return super().__new__(cls)

