#!/bin/bash
# Stop all real web applications

echo "Stopping real web applications..."

# Stop Juice Shop
if [ -f /tmp/juice-shop.pid ]; then
    PID=$(cat /tmp/juice-shop.pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        kill "$PID" 2>/dev/null || true
        echo "✓ Stopped Juice Shop"
    fi
    rm -f /tmp/juice-shop.pid
fi
pkill -f "juice-shop" 2>/dev/null && echo "✓ Cleaned up Juice Shop processes" || true

# Stop WebGoat
if [ -f /tmp/webgoat.pid ]; then
    PID=$(cat /tmp/webgoat.pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        kill "$PID" 2>/dev/null || true
        echo "✓ Stopped WebGoat"
    fi
    rm -f /tmp/webgoat.pid
fi
pkill -f "webgoat" 2>/dev/null && echo "✓ Cleaned up WebGoat processes" || true

# Stop DVWA
if [ -f /tmp/dvwa.pid ]; then
    PID=$(cat /tmp/dvwa.pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        kill "$PID" 2>/dev/null || true
        echo "✓ Stopped DVWA"
    fi
    rm -f /tmp/dvwa.pid
fi
pkill -f "dvwa" 2>/dev/null && echo "✓ Cleaned up DVWA processes" || true

echo "All applications stopped"
