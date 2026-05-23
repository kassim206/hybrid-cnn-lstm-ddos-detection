# Controlled hping3 Laboratory Procedure

This document describes the supplementary hping3 procedure for the controlled Docker/Kali laboratory environment.

## Purpose

The purpose of this procedure is to document how hping3 can be used inside an isolated local Docker network for academic testing. This procedure is not used to generate the main machine learning results. The main model evaluation is based on the CIC-DDoS2019 dataset.

## Safety Scope

This procedure must only be used inside the local Docker network created by this repository.

Do not use hping3 against:

- Public IP addresses
- University systems
- Third-party servers
- Production services
- Any system without explicit written permission

## Lab Components

| Component | Description |
|---|---|
| victim-web | Local Nginx web server container |
| kali-lab | Kali Linux container with hping3 installed |
| ddos_lab_net | Isolated Docker bridge network |

## Start the Lab

From the repository root:

```powershell
docker compose -f docker/docker-compose-lab.yml up --build