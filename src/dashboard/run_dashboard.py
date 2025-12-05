"""
Run the MedFedAgent Premium Dashboard

A modern, professional Flask-based dashboard for federated learning in healthcare.
"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.flask_app import app


def main():
    """Run the Flask dashboard."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   🏥 MedFedAgent Premium Dashboard                               ║
    ║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                        ║
    ║                                                                  ║
    ║   Privacy-Preserving Federated Learning for Healthcare           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    
    print(f"    🌐 Dashboard URL: http://{host}:{port}")
    print(f"    📊 Clinical View: http://{host}:{port}/clinical")
    print(f"    🔧 Technical View: http://{host}:{port}/technical")
    print(f"    🔒 Privacy View: http://{host}:{port}/privacy")
    print(f"    ⚖️  Fairness View: http://{host}:{port}/fairness")
    print()
    print("    Press Ctrl+C to stop the server")
    print()
    
    # Run the app
    app.run(
        host=host,
        port=port,
        debug=debug
    )


if __name__ == '__main__':
    main()
