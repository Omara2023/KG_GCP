import os

class EnvMixin:
    """Mixin to provide get_env method to other classes.""" 
    
    def _get_env(self, var_name: str) -> str:
        value = os.environ.get(var_name)
        if not value:
            raise ValueError(f"Missing required environment variable: {var_name}")
        return value