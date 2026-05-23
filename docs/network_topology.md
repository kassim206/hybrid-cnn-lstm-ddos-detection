# Controlled Docker Laboratory Network Topology

This document describes the supplementary controlled Docker-based laboratory environment added for reproducibility.

## Purpose

The Docker lab is used only for local, isolated, academic testing. It is not designed for public-network testing or unauthorized traffic generation.

## Components

| Component | Description |
|---|---|
| victim-web | Nginx web server used as a local victim service |
| kali-lab | Kali Linux container with hping3 installed |
| ddos_lab_net | Isolated Docker bridge network |

## Topology

```text
+---------------------+        Docker bridge network        +----------------------+
| kali_hping3_lab     |  ------------------------------->   | victim_web_server    |
| Kali Linux + hping3 |                                   | Nginx web server     |
+---------------------+                                   +----------------------+