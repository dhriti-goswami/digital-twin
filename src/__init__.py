"""
Diabetes Digital Twin - AI-powered diabetes management system.

This package provides:
- Multi-modal data ingestion (CGM, insulin, meals, activity)
- Physiologically-informed glucose prediction models (LSTM/Transformer with PINN)
- SHAP-based explainability for predictions
- LLM-powered conversational AI with medical guidelines RAG
- Adaptive learning with drift detection
- Real-time visualization dashboard

Usage:
    # As a Python module
    from src.digital_twin import DiabetesDigitalTwin
    twin = DiabetesDigitalTwin()
    twin.update_context(current_glucose=145)
    predictions = twin.predict()
    response = twin.chat("What will my glucose be in an hour?")

    # From command line
    python -m src.digital_twin
    python -m src.digital_twin --mode predict --glucose 145
    python -m src.digital_twin --mode server
"""

__version__ = "1.0.0"
__author__ = "Digital Twin Team"

# Lazy imports for better startup time
def get_digital_twin(**kwargs):
    """Factory function to create a DiabetesDigitalTwin instance."""
    from src.digital_twin import DiabetesDigitalTwin
    return DiabetesDigitalTwin(**kwargs)
