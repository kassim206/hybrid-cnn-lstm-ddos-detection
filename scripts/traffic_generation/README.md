# Traffic Generation Scripts

This folder contains supplementary local-only traffic generation material for the controlled Docker laboratory environment.

## Safety Notice

These scripts are only for the isolated Docker laboratory network created by this repository.

Do not use these scripts against:

- Public IP addresses
- University systems
- Third-party servers
- Production services
- Any system without explicit authorization

## Included Script

| File | Purpose |
|---|---|
| hping3_connectivity_test.sh | Performs a very small hping3 connectivity test inside the Docker lab |

## Docker Lab Startup

From the repository root:

```powershell
docker compose -f docker/docker-compose-lab.yml up --build