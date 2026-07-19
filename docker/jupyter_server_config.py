# JupyterLab configuration for the pose-dynamics Tier 0 container.
# Opening the server root lands directly on the quickstart notebook, with no login
# token (local use), so a non-programmer sees the running quickstart immediately.
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.root_dir = "/work"
# The Lab extension copies LabApp.default_url into ServerApp.default_url, so set it
# here (setting ServerApp.default_url alone is overridden back to /lab).
c.LabApp.default_url = "/lab/tree/notebooks/quickstart.ipynb"
c.ServerApp.default_url = "/lab/tree/notebooks/quickstart.ipynb"
c.ServerApp.allow_root = True
c.IdentityProvider.token = ""
c.PasswordIdentityProvider.hashed_password = ""
c.ServerApp.password = ""
