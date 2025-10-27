#!/usr/bin/env python3
"""
Simple server startup script that uses PowerShell commands
This is a fallback if npm is not found in PATH
"""

import subprocess
import sys
import os
import time
import threading
import signal
import webbrowser
from pathlib import Path

class SimpleServerManager:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
    def start_backend_powershell(self):
        """Start backend using PowerShell"""
        print("🚀 Starting Backend Server (PowerShell)...")
        try:
            # PowerShell command to start backend
            ps_command = '''
            cd backend
            python main.py
            '''
            
            self.backend_process = subprocess.Popen(
                ["powershell", "-Command", ps_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                print("✅ Backend server started successfully on http://localhost:8000")
                return True
            else:
                print("❌ Failed to start backend server")
                return False
                
        except Exception as e:
            print(f"❌ Error starting backend: {e}")
            return False
    
    def start_frontend_powershell(self):
        """Start frontend using PowerShell"""
        print("🚀 Starting Frontend Server (PowerShell)...")
        try:
            # PowerShell command to start frontend
            ps_command = '''
            npm start
            '''
            
            self.frontend_process = subprocess.Popen(
                ["powershell", "-Command", ps_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            time.sleep(8)
            
            if self.frontend_process.poll() is None:
                print("✅ Frontend server started successfully on http://localhost:3000")
                return True
            else:
                print("❌ Failed to start frontend server")
                return False
                
        except Exception as e:
            print(f"❌ Error starting frontend: {e}")
            return False
    
    def start_manual_instructions(self):
        """Provide manual instructions if automatic startup fails"""
        print("\n" + "="*60)
        print("🔧 MANUAL STARTUP INSTRUCTIONS")
        print("="*60)
        print("If automatic startup failed, please run these commands manually:")
        print()
        print("1. Open a new PowerShell window")
        print("2. Navigate to your project folder:")
        print("   cd C:\\Users\\ramva\\OneDrive\\Project")
        print()
        print("3. Start Backend (in first PowerShell window):")
        print("   cd backend")
        print("   python main.py")
        print()
        print("4. Open another PowerShell window")
        print("5. Start Frontend (in second PowerShell window):")
        print("   cd C:\\Users\\ramva\\OneDrive\\Project")
        print("   npm start")
        print()
        print("6. Open browser to: http://localhost:3000")
        print("="*60)
    
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
        print("🚀 BOOSTING ALGORITHMS DEMO - SIMPLE STARTUP")
        print("=" * 60)
        
        try:
            # Try PowerShell method
            print("💡 Trying PowerShell method...")
            
            # Start backend
            if not self.start_backend_powershell():
                print("❌ Backend failed. Showing manual instructions...")
                self.start_manual_instructions()
                return False
            
            # Start frontend
            if not self.start_frontend_powershell():
                print("❌ Frontend failed. Showing manual instructions...")
                self.start_manual_instructions()
                self.cleanup()
                return False
            
            # Open browser
            print("🌐 Opening browser...")
            time.sleep(2)
            try:
                webbrowser.open("http://localhost:3000")
                print("✅ Browser opened to http://localhost:3000")
            except Exception as e:
                print(f"⚠️ Could not open browser: {e}")
            
            print("\n" + "=" * 60)
            print("✅ SERVERS RUNNING SUCCESSFULLY!")
            print("🌐 Frontend: http://localhost:3000")
            print("🔧 Backend: http://localhost:8000")
            print("=" * 60)
            print("\n💡 Press Ctrl+C to stop all servers")
            print("=" * 60)
            
            # Keep running
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Received interrupt signal...")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            self.start_manual_instructions()
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
    manager = SimpleServerManager()
    
    # Set up signal handlers
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
