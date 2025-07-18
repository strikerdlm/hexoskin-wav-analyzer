# Jupyter Notebook Configuration for Valquiria Analysis
c = get_config()

# Set the IP address and port
c.NotebookApp.ip = '127.0.0.1'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = True
c.NotebookApp.notebook_dir = '.'

# Security settings
c.NotebookApp.token = ''
c.NotebookApp.password = ''
c.NotebookApp.allow_origin = '*'

# Enable extensions
c.NotebookApp.nbserver_extensions = {
    'jupyter_nbextensions_configurator': True,
}

# Kernel settings
c.KernelManager.autorestart = True
c.KernelManager.shutdown_wait_time = 10.0

# Content settings
c.ContentsManager.allow_hidden = True
c.ContentsManager.untitled_notebook = 'Untitled'
c.ContentsManager.untitled_file = 'untitled'
