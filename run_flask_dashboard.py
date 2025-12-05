"""
Run the MedFedAgent Premium Flask Dashboard

Usage:
    python run_flask_dashboard.py
    python run_flask_dashboard.py --port 8000
    python run_flask_dashboard.py --host 0.0.0.0 --port 5000
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dashboard.flask_app import app


def main():
    parser = argparse.ArgumentParser(description='Run MedFedAgent Premium Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')
    parser.add_argument('--no-debug', action='store_false', dest='debug', help='Disable debug mode')
    
    args = parser.parse_args()
    
    # Set environment for data paths
    os.environ.setdefault('RESULTS_DIR', './results')
    os.environ.setdefault('LOGS_DIR', './logs')
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║   🏥 MedFedAgent Premium Dashboard                                   ║
    ║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   ║
    ║                                                                      ║
    ║   Modern Flask-based dashboard for federated learning metrics        ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"    🌐 Starting server at: http://{args.host}:{args.port}")
    print()
    print("    📊 Available Pages:")
    print(f"       • Overview:   http://{args.host}:{args.port}/")
    print(f"       • Clinical:   http://{args.host}:{args.port}/clinical")
    print(f"       • Technical:  http://{args.host}:{args.port}/technical")
    print(f"       • Privacy:    http://{args.host}:{args.port}/privacy")
    print(f"       • Fairness:   http://{args.host}:{args.port}/fairness")
    print()
    print("    ⌨️  Keyboard Shortcuts:")
    print("       • T: Toggle dark/light theme")
    print("       • 1-5: Navigate to pages")
    print("       • Ctrl+R: Refresh data")
    print()
    print("    Press Ctrl+C to stop the server")
    print("    " + "═" * 68)
    print()
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
