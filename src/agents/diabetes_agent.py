"""
Diabetes Management AI Agent using LangChain and Ollama.

Provides a conversational interface that combines:
- Glucose prediction from PyTorch models
- SHAP-based explanations
- RAG for medical guidelines
- Personalized recommendations
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from src.agents.rag import MedicalGuidelinesRAG, setup_rag

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a highly specialized AI assistant for diabetes management. You help patients understand and manage their blood glucose levels safely.

IMPORTANT GUIDELINES:
1. Always prioritize patient safety
2. Never recommend stopping insulin without medical supervision
3. For severe hypoglycemia (<54 mg/dL) or hyperglycemia with ketones, advise seeking immediate medical care
4. Explain predictions in simple, understandable terms
5. Always provide context from medical guidelines when available
6. Be empathetic and supportive

You have access to the following tools:
{tools}

When responding:
1. If the patient asks about their glucose, use the prediction tools
2. Always explain WHY glucose is predicted to change using SHAP explanations
3. Reference medical guidelines when providing advice
4. If uncertain, recommend consulting their healthcare provider

Current patient context:
{patient_context}

Use the following format:
Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""


class DiabetesAgent:
    """
    AI Agent for diabetes management conversations.

    Integrates:
    - Glucose prediction model
    - SHAP explainer
    - Medical guidelines RAG
    - Conversational memory
    """

    def __init__(
        self,
        model_name: str = "llama3:8b",
        ollama_base_url: str = "http://localhost:11434",
        predictor=None,
        explainer=None,
        rag: Optional[MedicalGuidelinesRAG] = None,
        temperature: float = 0.7,
    ):
        if not OLLAMA_AVAILABLE:
            raise ImportError("LangChain is required. Install with: pip install langchain-community")

        self.predictor = predictor
        self.explainer = explainer
        self.rag = rag or setup_rag()

        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=temperature,
        )

        # Current patient context
        self.patient_context = {}
        self.current_glucose = None
        self.recent_data = {}

        # Simple message history
        self.message_history = []

        logger.info(f"DiabetesAgent initialized with model: {model_name}")

    def predict_glucose(self, horizon: str = "30") -> str:
        """
        Predict glucose at specified horizon.

        Args:
            horizon: Prediction horizon in minutes (30, 60, 90, or 120)
        """
        try:
            horizon_minutes = int(horizon)
            horizon_index = {30: 0, 60: 1, 90: 2, 120: 3}.get(horizon_minutes, 0)

            if self.predictor is None or self.current_glucose is None:
                # Return simulated prediction when model not available
                base = self.current_glucose or 120
                predicted = base + np.random.normal(0, 15)
                return f"Predicted glucose in {horizon_minutes} minutes: {predicted:.0f} mg/dL"

            # Use actual model prediction
            prediction, uncertainty = self.predictor.predict(self.recent_data["features"])
            predicted_value = prediction[0, horizon_index].item()
            uncertainty_value = uncertainty[0, horizon_index].item()

            return (
                f"Predicted glucose in {horizon_minutes} minutes: {predicted_value:.0f} mg/dL "
                f"(confidence interval: {predicted_value - 2*uncertainty_value:.0f} - {predicted_value + 2*uncertainty_value:.0f} mg/dL)"
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return f"Unable to make prediction: {str(e)}"

    def simulate_meal(self, carbs: str) -> str:
        """
        Simulate the effect of a meal on glucose.

        Args:
            carbs: Carbohydrates in grams
        """
        try:
            carb_grams = float(carbs)

            # Simple physiological model for meal response
            current = self.current_glucose or 120

            # Estimate peak glucose rise (roughly 3-4 mg/dL per gram of carbs without insulin)
            peak_rise = carb_grams * 3.5

            # With proper insulin coverage, should be reduced
            expected_peak = current + (peak_rise * 0.4)  # Assuming partial coverage
            peak_time = 60  # Peak typically at 60 minutes

            # Time to return to baseline
            return_time = 180  # 3 hours

            result = f"""
**Meal Simulation: {carb_grams:.0f}g carbohydrates**

Starting glucose: {current:.0f} mg/dL

Predicted response:
- Peak glucose: ~{expected_peak:.0f} mg/dL (around {peak_time} minutes after eating)
- Return to baseline: approximately {return_time} minutes

**Recommendation:**
"""

            if expected_peak > 180:
                result += f"- Consider pre-bolusing 15-20 minutes before eating\n"
                result += f"- Suggested insulin: {carb_grams / 10:.1f} units (assuming 1:10 ratio)\n"
            else:
                result += f"- Glucose should stay in target range with proper bolus timing\n"

            return result

        except Exception as e:
            return f"Unable to simulate meal: {str(e)}"

    def explain_prediction(self, placeholder: str = "") -> str:
        """Get explanation for the most recent prediction."""
        if self.explainer is None:
            return self._generate_simple_explanation()

        try:
            explanation = self.explainer.explain_prediction(
                self.recent_data["features"],
                horizon_index=0
            )
            return explanation["explanation"]
        except Exception as e:
            logger.error(f"Explanation error: {e}")
            return self._generate_simple_explanation()

    def _generate_simple_explanation(self) -> str:
        """Generate a simple explanation when SHAP is not available."""
        current = self.current_glucose or 120
        factors = []

        if self.recent_data.get("recent_carbs", 0) > 20:
            factors.append(f"Recent carbohydrate intake ({self.recent_data.get('recent_carbs', 0):.0f}g) is raising glucose")

        if self.recent_data.get("iob", 0) > 1:
            factors.append(f"Active insulin ({self.recent_data.get('iob', 0):.1f} units) is working to lower glucose")

        if self.recent_data.get("recent_exercise", False):
            factors.append("Recent exercise is increasing insulin sensitivity")

        hour = datetime.now().hour
        if 5 <= hour <= 8:
            factors.append("Dawn phenomenon may be causing morning glucose rise")

        if not factors:
            factors.append("Glucose appears stable with no major influencing factors")

        explanation = "**Key factors affecting your glucose:**\n\n"
        for i, factor in enumerate(factors, 1):
            explanation += f"{i}. {factor}\n"

        return explanation

    def search_guidelines(self, query: str) -> str:
        """Search medical guidelines."""
        try:
            results = self.rag.search(query, n_results=3)

            if not results:
                return "No specific guidelines found for this query."

            response = "**Relevant Medical Guidelines:**\n\n"
            for result in results:
                response += f"### {result['title']}\n"
                response += f"{result['content'][:500]}...\n\n"

            return response
        except Exception as e:
            logger.error(f"Guidelines search error: {e}")
            return f"Unable to search guidelines: {str(e)}"

    def get_patient_summary(self, placeholder: str = "") -> str:
        """Get patient glucose summary."""
        summary = "**Patient Summary:**\n\n"

        if self.current_glucose:
            summary += f"- Current glucose: {self.current_glucose:.0f} mg/dL\n"

            if self.current_glucose < 70:
                summary += "- Status: ⚠️ LOW - Consider treating with fast-acting carbs\n"
            elif self.current_glucose > 180:
                summary += "- Status: ⚠️ HIGH - Monitor and consider correction if needed\n"
            else:
                summary += "- Status: ✅ In target range\n"

        if self.recent_data.get("tir"):
            summary += f"- Time in Range (24h): {self.recent_data['tir']:.1f}%\n"

        if self.recent_data.get("avg_glucose"):
            summary += f"- Average glucose (24h): {self.recent_data['avg_glucose']:.0f} mg/dL\n"

        return summary

    def update_context(
        self,
        current_glucose: float,
        recent_cgm: Optional[np.ndarray] = None,
        recent_insulin: Optional[list] = None,
        recent_meals: Optional[list] = None,
        patient_info: Optional[dict] = None,
    ):
        """Update the agent's context with latest patient data."""
        self.current_glucose = current_glucose

        self.recent_data = {
            "current_glucose": current_glucose,
            "recent_carbs": sum(m.get("carbs", 0) for m in (recent_meals or [])),
            "iob": sum(i.get("units", 0) * 0.8 for i in (recent_insulin or [])),  # Simplified IOB
            "recent_exercise": False,  # Would come from activity data
        }

        if patient_info:
            self.patient_context = patient_info

        if recent_cgm is not None:
            # Convert to numpy array if needed
            cgm_array = np.array(recent_cgm) if not isinstance(recent_cgm, np.ndarray) else recent_cgm
            # Calculate stats
            self.recent_data["avg_glucose"] = np.mean(cgm_array)
            self.recent_data["tir"] = np.mean((cgm_array >= 70) & (cgm_array <= 180)) * 100

    def chat(self, message: str) -> str:
        """
        Process a chat message and return response from Ollama.

        Args:
            message: User message

        Returns:
            AI response
        """
        try:
            # Build context string
            context = self._build_context()

            # Check for urgent situations first
            urgent_response = self._check_urgent_situations(message)
            if urgent_response:
                return urgent_response

            # Format prompt with full context
            prompt = f"""You are a diabetes management AI assistant. Be helpful, safe, and empathetic.

Patient Context:
{context}

Recent Messages:
{self._format_message_history()}

User: {message}

Provide a helpful response. If discussing glucose predictions or management, be specific and reference relevant guidelines. Always prioritize safety. Be concise and friendly."""

            # Get response from Ollama
            response = self.llm.invoke(prompt)

            # Store in history (keep last 10 messages)
            self.message_history.append({"role": "user", "content": message})
            self.message_history.append({"role": "assistant", "content": response})
            self.message_history = self.message_history[-20:]  # Keep last 10 turns

            return response

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}. Please try again or consult your healthcare provider for immediate concerns."

    def _format_message_history(self) -> str:
        """Format message history for context."""
        if not self.message_history:
            return "No previous conversation"

        recent = self.message_history[-6:]  # Last 3 turns
        formatted = []
        for msg in recent:
            role_indicator = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:150]
            formatted.append(f"{role_indicator}: {content}")

        return "\n".join(formatted)

    def _build_context(self) -> str:
        """Build context string from current patient data."""
        parts = []

        if self.current_glucose:
            parts.append(f"Current glucose: {self.current_glucose:.0f} mg/dL")

            if self.current_glucose < 70:
                parts.append("⚠️ PATIENT IS HYPOGLYCEMIC - Prioritize safety guidance")
            elif self.current_glucose > 250:
                parts.append("⚠️ PATIENT IS VERY HIGH - Check for ketones")

        if self.recent_data.get("iob"):
            parts.append(f"Insulin on board: ~{self.recent_data['iob']:.1f} units")

        if self.recent_data.get("recent_carbs"):
            parts.append(f"Recent carbs: {self.recent_data['recent_carbs']:.0f}g")

        if self.recent_data.get("tir"):
            parts.append(f"Time in Range (24h): {self.recent_data['tir']:.1f}%")

        if self.recent_data.get("avg_glucose"):
            parts.append(f"Average glucose (24h): {self.recent_data['avg_glucose']:.0f} mg/dL")

        return "\n".join(parts) if parts else "No current patient data available"

    def _check_urgent_situations(self, message: str) -> Optional[str]:
        """Check for urgent situations that need immediate response."""
        message_lower = message.lower()

        # Check for severe hypoglycemia
        if self.current_glucose and self.current_glucose < 54:
            return """🚨 **URGENT: Severe Low Blood Sugar Detected**

Your glucose is critically low at {:.0f} mg/dL.

**Take these steps immediately:**
1. Consume 15-20g of fast-acting carbohydrates NOW (glucose tablets, juice, or regular soda)
2. Do NOT take insulin
3. Sit or lie down if you feel unsteady
4. Recheck glucose in 15 minutes
5. If symptoms don't improve or you feel confused, have someone call for emergency help

If you cannot treat yourself, this is a medical emergency requiring glucagon or emergency services.

Are you able to treat yourself right now?""".format(self.current_glucose)

        # Check for DKA warning signs in message
        dka_keywords = ["ketones", "vomiting", "fruity breath", "confused", "sick", "nausea"]
        if self.current_glucose and self.current_glucose > 250:
            if any(keyword in message_lower for keyword in dka_keywords):
                return """🚨 **URGENT: Possible DKA Warning Signs**

With high glucose ({:.0f} mg/dL) and the symptoms you describe, you may be at risk for Diabetic Ketoacidosis (DKA).

**Seek immediate medical attention if you have:**
- Moderate or large ketones
- Persistent vomiting for more than 2 hours
- Fruity breath odor
- Rapid breathing
- Confusion or drowsiness

**Do NOT wait** - DKA can become life-threatening quickly.

Please contact your healthcare provider or go to the emergency room immediately.

Is there someone who can take you to get medical care?""".format(self.current_glucose)

        return None


def create_diabetes_agent(
    predictor=None,
    explainer=None,
    model_name: str = "llama3:8b",
) -> DiabetesAgent:
    """
    Factory function to create a diabetes management agent.

    Args:
        predictor: Trained GlucosePredictor model
        explainer: GlucoseExplainer instance
        model_name: Ollama model to use

    Returns:
        Configured DiabetesAgent
    """
    try:
        rag = setup_rag()
    except Exception as e:
        logger.warning(f"RAG initialization failed: {e}. Agent will work without medical guidelines search.")
        rag = None

    return DiabetesAgent(
        model_name=model_name,
        predictor=predictor,
        explainer=explainer,
        rag=rag,
    )
