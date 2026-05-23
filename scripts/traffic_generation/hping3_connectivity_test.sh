#!/bin/bash

echo "Controlled local hping3 connectivity test"
echo "Target: victim-web Docker service"
echo "This script sends only a very small number of packets for lab verification."

TARGET="victim-web"

echo "Resolving target..."
ping -c 2 $TARGET

echo "Running small hping3 TCP connectivity test..."
hping3 -S -p 80 -c 5 $TARGET

echo "Test completed."