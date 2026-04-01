# Security Policy

## Reporting a Vulnerability

If you find a security vulnerability in docker_mlx_cpp, please report it responsibly:

**Email:** dev@robotflowlabs.com

**What to include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact

**Response time:** We will acknowledge within 48 hours and provide a fix timeline within 7 days.

## Scope

docker_mlx_cpp runs a daemon on the host with Metal GPU access. Security considerations:

- The MLX Daemon listens on `localhost:12435` by default (not exposed externally)
- The Docker gateway runs inside the Docker network
- No authentication is required by default (designed for local development)
- For production use, add authentication middleware to the gateway

## Not in Scope

- Vulnerabilities in upstream dependencies (MLX, mlx-lm, FastAPI, etc.) — report to their maintainers
- Issues requiring physical access to the machine
