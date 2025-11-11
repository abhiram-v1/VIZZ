#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated server startup script for Boosting Algorithms Demo
This script automatically starts both backend and frontend servers.
"""

import subprocess
import sys
import os
import time
import threading
import signal
import webbrowser
from pathlib import Path

# Fix Windows console encoding for emoji support
def safe_print(text):
    """Safely print text, handling encoding errors on Windows"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace emojis with ASCII alternatives for Windows
        replacements = {
            '🚀': '[START]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '💡': '[TIP]',
            '📦': '[INSTALL]',
            '⏳': '[WAIT]',
            '🌐': '[WEB]',
            '🛑': '[STOP]',
            '📊': '[MONITOR]',
            '👋': '[BYE]',
        }
        safe_text = text
        for emoji, replacement in replacements.items():
            safe_text = safe_text.replace(emoji, replacement)
        print(safe_text)

if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Fallback for older Python versions
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
else:
    # On non-Windows, use regular print
    safe_print = print

class ServerManager:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
    def start_backend(self):
        """Start the FastAPI backend server"""
        safe_print("🚀 Starting Backend Server...")
        try:
            # Change to backend directory and start the server
            backend_dir = Path("backend")
            if not backend_dir.exists():
                safe_print("❌ Backend directory not found!")
                return False
                
            # Start backend server
            self.backend_process = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Check if backend is running
            if self.backend_process.poll() is None:
                safe_print("✅ Backend server started successfully on http://localhost:8000")
                return True
            else:
                safe_print("❌ Failed to start backend server")
                return False
                
        except Exception as e:
            safe_print(f"❌ Error starting backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the React frontend server"""
        safe_print("🚀 Starting Frontend Server...")
        try:
            # Check if package.json exists
            if not Path("package.json").exists():
                safe_print("❌ package.json not found! Make sure you're in the project root.")
                return False
            
            # Check if npm is available
            try:
                subprocess.run(["npm", "--version"], capture_output=True, check=True)
                safe_print("✅ npm found in PATH")
            except (subprocess.CalledProcessError, FileNotFoundError):
                safe_print("❌ npm not found in PATH")
                safe_print("💡 Trying alternative methods...")
                
                # Try common npm locations
                npm_paths = [
                    "C:\\Program Files\\nodejs\\npm.cmd",
                    "C:\\Program Files (x86)\\nodejs\\npm.cmd",
                    "C:\\Users\\{}\\AppData\\Roaming\\npm\\npm.cmd".format(os.getenv('USERNAME', '')),
                    "npm.cmd",
                    "npm"
                ]
                
                npm_cmd = None
                for path in npm_paths:
                    try:
                        subprocess.run([path, "--version"], capture_output=True, check=True)
                        npm_cmd = path
                        safe_print(f"✅ Found npm at: {path}")
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                
                if not npm_cmd:
                    safe_print("❌ Could not find npm anywhere!")
                    safe_print("💡 Please install Node.js from https://nodejs.org/")
                    safe_print("💡 Or try running: npm --version")
                    return False
            
            # Install dependencies if node_modules doesn't exist
            if not Path("node_modules").exists():
                safe_print("📦 Installing frontend dependencies...")
                try:
                    install_process = subprocess.run(
                        [npm_cmd or "npm", "install"],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    if install_process.returncode != 0:
                        safe_print("❌ Failed to install dependencies")
                        print(install_process.stderr)
                        return False
                    safe_print("✅ Dependencies installed successfully")
                except subprocess.TimeoutExpired:
                    safe_print("❌ Installation timed out")
                    return False
                except Exception as e:
                    safe_print(f"❌ Installation error: {e}")
                    return False
            
            # Start frontend server
            safe_print("🚀 Starting React development server...")
            self.frontend_process = subprocess.Popen(
                [npm_cmd or "npm", "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for frontend to start
            safe_print("⏳ Waiting for frontend to start...")
            time.sleep(8)  # Give more time for React to start
            
            if self.frontend_process.poll() is None:
                safe_print("✅ Frontend server started successfully on http://localhost:3000")
                return True
            else:
                safe_print("❌ Failed to start frontend server")
                # Get error output
                stdout, stderr = self.frontend_process.communicate()
                if stderr:
                    print(f"Error output: {stderr}")
                return False
                
        except Exception as e:
            safe_print(f"❌ Error starting frontend: {e}")
            return False
    
    def monitor_servers(self):
        """Monitor server output and status"""
        def monitor_backend():
            if self.backend_process:
                for line in iter(self.backend_process.stdout.readline, ''):
                    if line and self.running:
                        print(f"[BACKEND] {line.strip()}")
        
        def monitor_frontend():
            if self.frontend_process:
                for line in iter(self.frontend_process.stdout.readline, ''):
                    if line and self.running:
                        print(f"[FRONTEND] {line.strip()}")
        
        # Start monitoring threads
        backend_thread = threading.Thread(target=monitor_backend, daemon=True)
        frontend_thread = threading.Thread(target=monitor_frontend, daemon=True)
        
        backend_thread.start()
        frontend_thread.start()
        
        return backend_thread, frontend_thread
    
    def open_browser(self):
        """Open browser to the application"""
        safe_print("🌐 Opening browser...")
        time.sleep(2)  # Wait for servers to be ready
        try:
            webbrowser.open("http://localhost:3000")
            safe_print("✅ Browser opened to http://localhost:3000")
        except Exception as e:
            safe_print(f"⚠️ Could not open browser automatically: {e}")
            safe_print("🌐 Please manually open http://localhost:3000 in your browser")
    
    def cleanup(self):
        """Clean up processes"""
        safe_print("\n🛑 Shutting down servers...")
        self.running = False
        
        if self.backend_process:
            self.backend_process.terminate()
            safe_print("✅ Backend server stopped")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            safe_print("✅ Frontend server stopped")
    
    def run(self):
        """Main execution function"""
        print("=" * 60)
        safe_print("🚀 BOOSTING ALGORITHMS DEMO - SERVER STARTUP")
        print("=" * 60)
        
        try:
            # Start backend
            if not self.start_backend():
                safe_print("❌ Failed to start backend. Exiting...")
                return False
            
            # Start frontend
            if not self.start_frontend():
                safe_print("❌ Failed to start frontend. Exiting...")
                self.cleanup()
                return False
            
            # Start monitoring
            safe_print("\n📊 Monitoring server output...")
            self.monitor_servers()
            
            # Open browser
            self.open_browser()
            
            print("\n" + "=" * 60)
            safe_print("✅ SERVERS RUNNING SUCCESSFULLY!")
            safe_print("🌐 Frontend: http://localhost:3000")
            print("🔧 Backend: http://localhost:8000")
            print("📚 API Docs: http://localhost:8000/docs")
            print("=" * 60)
            safe_print("\n💡 Press Ctrl+C to stop all servers")
            print("=" * 60)
            
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            safe_print("\n\n🛑 Received interrupt signal...")
        except Exception as e:
            safe_print(f"\n❌ Unexpected error: {e}")
        finally:
            self.cleanup()
            safe_print("\n👋 Goodbye!")
        
        return True

def main():
    """Main function"""
    # Check if we're in the right directory
    if not Path("package.json").exists() or not Path("backend").exists():
        safe_print("❌ Error: Please run this script from the project root directory")
        print("   Make sure both package.json and backend/ folder exist")
        sys.exit(1)
    
    # Create server manager and run
    manager = ServerManager()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        safe_print(f"\n🛑 Received signal {signum}")
        manager.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the servers
    success = manager.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
