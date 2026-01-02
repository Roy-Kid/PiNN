import os

def get_backend(params=None) -> str:
    # Priority: explicit params > env var > default
    if params and "backend" in params and params["backend"]:
        return str(params["backend"]).lower()
    return os.environ.get("PINN_BACKEND", "tf").lower()