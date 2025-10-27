#!/usr/bin/env python3
"""
Launch the animated decision boundary experiment
Generates data and opens the web visualization
"""

import subprocess
import webbrowser
import os
import sys
from pathlib import Path

def main():
    print("🧠 ANIMATED DECISION BOUNDARY EXPERIMENT")
    print("="*50)
    
    # Check if we're in the right directory
    if not Path("enhanced_animated_boundary.html").exists():
        print("❌ enhanced_animated_boundary.html not found!")
        print("💡 Make sure you're in the project directory")
        return
    
    print("🚀 Starting animated decision boundary visualization...")
    
    # Generate boundary data (optional)
    if Path("generate_boundary_data.py").exists():
        print("📊 Generating boundary data...")
        try:
            subprocess.run([sys.executable, "generate_boundary_data.py"], check=True)
            print("✅ Boundary data generated")
        except subprocess.CalledProcessError:
            print("⚠️ Could not generate boundary data, using default data")
    else:
        print("⚠️ generate_boundary_data.py not found, using default data")
    
    # Open the web visualization
    html_file = Path("enhanced_animated_boundary.html").absolute()
    print(f"🌐 Opening web visualization: {html_file}")
    
    try:
        webbrowser.open(f"file://{html_file}")
        print("✅ Web visualization opened in browser")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
        print(f"💡 Please manually open: {html_file}")
    
    print("\n🎯 EXPERIMENT READY!")
    print("="*30)
    print("📊 What you'll see:")
    print("  • Red dots: Stroke patients")
    print("  • Green dots: No stroke patients") 
    print("  • Animated line: Decision boundary")
    print("  • Learning process: Watch accuracy improve")
    print("")
    print("🎮 Controls:")
    print("  • Play/Pause: Start/stop animation")
    print("  • Next/Previous: Step through iterations")
    print("  • Slow Motion: Watch in detail")
    print("  • Reset: Start over")
    print("")
    print("🔬 The Science:")
    print("  • Iteration 1: Simple boundary (~65% accuracy)")
    print("  • Iteration 2-3: Learning from mistakes (~75-85%)")
    print("  • Iteration 4-5: Refined boundary (~90-95%)")
    print("")
    print("✅ Experiment completed! Check your browser.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Experiment cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure all files are in the same directory")
