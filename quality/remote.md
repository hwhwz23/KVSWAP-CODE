## Remote access setup

This guide configures Jupyter on the device and exposes it through **frpc** so one can connect remotely.

### 1. Jupyter configuration

Generate a config file (if you do not already have one):

```bash
jupyter notebook --generate-config
```

Create a password hash:

```bash
python -c "from jupyter_server.auth import passwd; print(passwd())"
```

Edit `~/.jupyter/jupyter_notebook_config.py` and set at least:

```python
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.password = 'argon2:...'   # paste the hash from passwd()
c.ServerApp.allow_remote_access = True
```

### 2. Start Jupyter (local)

Run Jupyter Lab in the background (adjust paths and port if needed):

```bash
nohup jupyter lab --no-browser --ip=127.0.0.1 --port=8888 > jupyter.log 2>&1 &
```

You can then open the UI in a browser on the machine itself.

### 3. Expose via frpc (remote access)

To allow access from outside the LAN, run **frpc** with your own `frpc.toml` (download the matching **frpc** binary first).

Example (paths may differ):

```bash
nohup ./frpc/frp_0.61.1_linux_arm64/frpc -c ./frpc/frp_0.61.1_linux_arm64/frpc.toml > frpc.log 2>&1 &
```

After the tunnel is up, use the URL and credentials you configured on the frp server side.

