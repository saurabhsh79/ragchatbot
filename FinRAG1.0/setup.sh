#!/bin/bash
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0
pip install --upgrade pip
echo "✅ System setup complete."
