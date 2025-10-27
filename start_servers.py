#!/usr/bin/env python3
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

class ServerManager:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
    def start_backend(self):
        """Start the FastAPI backend server"""
        print("🚀 Starting Backend Server...")
        try:
            # Change to backend directory and start the server
            backend_dir = Path("backend")
            if not backend_dir.exists():
                print("❌ Backend directory not found!")
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
                print("✅ Backend server started successfully on http://localhost:8000")
                return True
            else:
                print("❌ Failed to start backend server")
                return False
                
        except Exception as e:
            print(f"❌ Error starting backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the React frontend server"""
        print("🚀 Starting Frontend Server...")
        try:
            # Check if package.json exists
            if not Path("package.json").exists():
                print("❌ package.json not found! Make sure you're in the project root.")
                return False
            
            # Check if npm is available
            try:
                subprocess.run(["npm", "--version"], capture_output=True, check=True)
                print("✅ npm found in PATH")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ npm not found in PATH")
                print("💡 Trying alternative methods...")
                
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
                        print(f"✅ Found npm at: {path}")
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                
                if not npm_cmd:
                    print("❌ Could not find npm anywhere!")
                    print("💡 Please install Node.js from https://nodejs.org/")
                    print("💡 Or try running: npm --version")
                    return False
            
            # Install dependencies if node_modules doesn't exist
            if not Path("node_modules").exists():
                print("📦 Installing frontend dependencies...")
                try:
                    install_process = subprocess.run(
                        [npm_cmd or "npm", "install"],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    if install_process.returncode != 0:
                        print("❌ Failed to install dependencies")
                        print(install_process.stderr)
                        return False
                    print("✅ Dependencies installed successfully")
                except subprocess.TimeoutExpired:
                    print("❌ Installation timed out")
                    return False
                except Exception as e:
                    print(f"❌ Installation error: {e}")
                    return False
            
            # Start frontend server
            print("🚀 Starting React development server...")
            self.frontend_process = subprocess.Popen(
                [npm_cmd or "npm", "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for frontend to start
            print("⏳ Waiting for frontend to start...")
            time.sleep(8)  # Give more time for React to start
            
            if self.frontend_process.poll() is None:
                print("✅ Frontend server started successfully on http://localhost:3000")
                return True
            else:
                print("❌ Failed to start frontend server")
                # Get error output
                stdout, stderr = self.frontend_process.communicate()
                if stderr:
                    print(f"Error output: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting frontend: {e}")
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
        print("🌐 Opening browser...")
        time.sleep(2)  # Wait for servers to be ready
        try:
            webbrowser.open("http://localhost:3000")
            print("✅ Browser opened to http://localhost:3000")
        except Exception as e:
            print(f"⚠️ Could not open browser automatically: {e}")
            print("🌐 Please manually open http://localhost:3000 in your browser")
    
    def cleanup(self):
        """Clean up processes"""
        print("\n🛑 Shutting down servers...")
        self.running = False
        
        if self.backend_process:
            self.backend_process.terminate()
            print("✅ Backend server stopped")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            print("✅ Frontend server stopped")
    
    def run(self):
        """Main execution function"""
        print("=" * 60)
        print("🚀 BOOSTING ALGORITHMS DEMO - SERVER STARTUP")
        print("=" * 60)
        
        try:
            # Start backend
            if not self.start_backend():
                print("❌ Failed to start backend. Exiting...")
                return False
            
            # Start frontend
            if not self.start_frontend():
                print("❌ Failed to start frontend. Exiting...")
                self.cleanup()
                return False
            
            # Start monitoring
            print("\n📊 Monitoring server output...")
            self.monitor_servers()
            
            # Open browser
            self.open_browser()
            
            print("\n" + "=" * 60)
            print("✅ SERVERS RUNNING SUCCESSFULLY!")
            print("🌐 Frontend: http://localhost:3000")
            print("🔧 Backend: http://localhost:8000")
            print("📚 API Docs: http://localhost:8000/docs")
            print("=" * 60)
            print("\n💡 Press Ctrl+C to stop all servers")
            print("=" * 60)
            
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Received interrupt signal...")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            self.cleanup()
            print("\n👋 Goodbye!")
        
        return True

def main():
    """Main function"""
    # Check if we're in the right directory
    if not Path("package.json").exists() or not Path("backend").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Make sure both package.json and backend/ folder exist")
        sys.exit(1)
    
    # Create server manager and run
    manager = ServerManager()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        manager.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the servers
    success = manager.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
