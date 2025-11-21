#!/bin/bash

# Quick Setup Script for Multilingual IR System

echo "=========================================="
echo "Multilingual IR System - Quick Setup"
echo "=========================================="

# Check Python version
echo -e "\n1. Checking Python version..."
python --version

# Create virtual environment
echo -e "\n2. Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo -e "\n3. Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo -e "\n4. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "\n=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo -e "\nNext steps:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Build the index: python main.py build --sample-size 5000"
echo "  3. Run a search: python main.py search 'your query here'"
echo "  4. Or use interactive mode: python main.py interactive"
echo ""
