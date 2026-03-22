"""
RAG (Retrieval-Augmented Generation) module for medical guidelines.

Uses ChromaDB for vector storage and retrieval of diabetes management guidelines
to ensure medically safe and accurate LLM responses.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Comprehensive diabetes management guidelines based on ADA Standards of Care
# These are real clinical guidelines used for diabetes management
DIABETES_GUIDELINES = [
    # Hypoglycemia Management
    {
        "id": "hypo_1",
        "category": "hypoglycemia",
        "title": "Hypoglycemia Definition and Classification",
        "content": """
        Hypoglycemia is classified as:
        - Level 1 (Alert): Glucose <70 mg/dL (3.9 mmol/L) but ≥54 mg/dL (3.0 mmol/L). Treat with fast-acting carbohydrates.
        - Level 2 (Serious): Glucose <54 mg/dL (3.0 mmol/L). Requires immediate treatment. Associated with increased mortality risk.
        - Level 3 (Severe): Any glucose level with altered mental and/or physical status requiring assistance for treatment.

        Treatment: Rule of 15 - consume 15-20g fast-acting carbohydrates, wait 15 minutes, recheck glucose. Repeat if still <70 mg/dL.
        """
    },
    {
        "id": "hypo_2",
        "category": "hypoglycemia",
        "title": "Hypoglycemia Prevention",
        "content": """
        To prevent hypoglycemia:
        1. Monitor glucose regularly, especially before driving, exercise, and sleep
        2. Do not skip meals
        3. Adjust insulin doses based on carbohydrate intake and activity
        4. Carry fast-acting glucose at all times (glucose tablets, juice)
        5. Use CGM with alerts set appropriately
        6. Consider lower overnight basal rates during increased activity days
        7. Have glucagon available for severe hypoglycemia emergencies
        """
    },

    # Hyperglycemia Management
    {
        "id": "hyper_1",
        "category": "hyperglycemia",
        "title": "Hyperglycemia Classification and Targets",
        "content": """
        Glucose targets for most adults with diabetes:
        - Fasting/Pre-meal: 80-130 mg/dL (4.4-7.2 mmol/L)
        - Post-meal peak (1-2 hours): <180 mg/dL (<10.0 mmol/L)
        - Time in Range (70-180 mg/dL): >70%
        - Time below Range (<70 mg/dL): <4%
        - Time above Range (>180 mg/dL): <25%

        Hyperglycemia >250 mg/dL requires assessment for ketones in Type 1 diabetes.
        Persistent hyperglycemia >300 mg/dL requires medical attention.
        """
    },
    {
        "id": "hyper_2",
        "category": "hyperglycemia",
        "title": "Correction Dose Guidelines",
        "content": """
        Correction doses should be calculated using:
        Correction Factor (CF) = 1800 / Total Daily Insulin (for rapid-acting insulin)

        Example: If TDD = 50 units, CF = 1800/50 = 36
        This means 1 unit of rapid-acting insulin will lower glucose by approximately 36 mg/dL.

        Correction Dose = (Current Glucose - Target Glucose) / Correction Factor

        Important: Allow 2-3 hours between correction doses to prevent insulin stacking.
        Do not correct if active insulin on board (IOB) exceeds correction dose.
        """
    },

    # Meal and Carbohydrate Management
    {
        "id": "meal_1",
        "category": "meals",
        "title": "Carbohydrate Counting and Bolus Calculation",
        "content": """
        Meal bolus calculation:
        1. Count total carbohydrates in the meal
        2. Divide by Insulin-to-Carb Ratio (ICR)
        3. ICR typically ranges from 1:5 to 1:20 (1 unit per X grams of carbs)
        4. ICR = 500 / Total Daily Insulin (approximate starting point)

        Timing: Rapid-acting insulin should be given 15-20 minutes before eating.
        For high-glycemic meals, consider giving bolus earlier.
        For high-fat/protein meals, consider extended bolus (pump) or split dose.
        """
    },
    {
        "id": "meal_2",
        "category": "meals",
        "title": "Glycemic Index Considerations",
        "content": """
        Glycemic Index affects glucose rise:
        - High GI (>70): White bread, rice, potatoes - rapid glucose spike
        - Medium GI (55-70): Whole grains, sweet potatoes
        - Low GI (<55): Most vegetables, legumes, whole fruits

        High GI foods cause faster, higher glucose spikes requiring earlier insulin timing.
        High fat/protein meals delay glucose absorption - consider this when dosing.
        Fiber slows carbohydrate absorption.
        """
    },

    # Exercise and Activity
    {
        "id": "exercise_1",
        "category": "exercise",
        "title": "Exercise and Glucose Management",
        "content": """
        Exercise effects on glucose:
        - Aerobic exercise typically lowers glucose
        - High-intensity/anaerobic exercise may increase glucose initially
        - Effects can last 24-48 hours post-exercise

        Guidelines:
        1. Check glucose before exercise: if <90 mg/dL, consume 15-30g carbs
        2. If >250 mg/dL with ketones, do not exercise
        3. Reduce insulin by 20-50% for exercise depending on duration/intensity
        4. Monitor for delayed hypoglycemia (up to 24 hours post-exercise)
        5. Keep fast-acting carbs readily available during exercise
        """
    },
    {
        "id": "exercise_2",
        "category": "exercise",
        "title": "Exercise Insulin Adjustments",
        "content": """
        Insulin adjustments for exercise:
        - Light exercise (walking): Reduce basal by 20%, bolus by 25%
        - Moderate exercise (jogging, swimming): Reduce basal by 30-40%, bolus by 50%
        - Vigorous exercise (competitive sports): Reduce basal by 50%, bolus by 50-75%

        For pumps: Consider temp basal 1-2 hours before exercise.
        For MDI: Consider reducing morning long-acting dose on active days.
        Always carry glucose and have a post-exercise snack plan.
        """
    },

    # Sick Day Management
    {
        "id": "sick_1",
        "category": "sick_day",
        "title": "Sick Day Rules",
        "content": """
        During illness:
        1. NEVER stop insulin, even if not eating
        2. Check glucose every 2-4 hours
        3. Check ketones if glucose >250 mg/dL
        4. Stay hydrated with sugar-free fluids
        5. If vomiting or ketones present, contact healthcare provider

        May need MORE insulin during illness due to stress hormones.
        Warning signs requiring immediate medical attention:
        - Moderate/large ketones
        - Persistent vomiting (>4 hours)
        - Glucose >300 mg/dL despite extra insulin
        - Signs of dehydration
        """
    },

    # Technology and CGM
    {
        "id": "tech_1",
        "category": "technology",
        "title": "CGM Use and Interpretation",
        "content": """
        CGM Best Practices:
        1. Calibrate per manufacturer instructions
        2. Use trend arrows to anticipate glucose changes
        3. Set alerts: Low urgent ≤55, Low 70, High 180, High urgent ≥250
        4. CGM may lag finger stick by 5-15 minutes

        Trend Arrow Interpretation (approximate mg/dL per minute):
        ↑↑ Rising rapidly (>3 mg/dL/min)
        ↑ Rising (2-3 mg/dL/min)
        → Stable
        ↓ Falling (2-3 mg/dL/min)
        ↓↓ Falling rapidly (>3 mg/dL/min)
        """
    },

    # Time in Range
    {
        "id": "tir_1",
        "category": "targets",
        "title": "Time in Range Goals",
        "content": """
        Recommended Time in Range targets (ADA/EASD consensuwhats):

        General Type 1/Type 2:
        - TIR (70-180 mg/dL): >70%
        - TBR (<70 mg/dL): <4%
        - TBR (<54 mg/dL): <1%
        - TAR (>180 mg/dL): <25%
        - TAR (>250 mg/dL): <5%

        Older adults/high risk:
        - TIR (70-180 mg/dL): >50%
        - TBR (<70 mg/dL): <1%

        Each 5% increase in TIR is clinically significant.
        """
    },

    # Insulin Pharmacokinetics
    {
        "id": "insulin_1",
        "category": "insulin",
        "title": "Insulin Action Profiles",
        "content": """
        Insulin action times:
        Rapid-acting (Humalog, Novolog, Apidra):
        - Onset: 10-20 minutes
        - Peak: 1-2 hours
        - Duration: 3-5 hours

        Short-acting (Regular):
        - Onset: 30-60 minutes
        - Peak: 2-4 hours
        - Duration: 5-8 hours

        Long-acting (Lantus, Levemir, Tresiba):
        - Onset: 1-2 hours
        - Peak: Relatively flat
        - Duration: 18-24+ hours

        Consider these profiles when timing doses and predicting glucose changes.
        """
    },

    # Dawn Phenomenon
    {
        "id": "dawn_1",
        "category": "patterns",
        "title": "Dawn Phenomenon and Somogyi Effect",
        "content": """
        Dawn Phenomenon:
        - Natural rise in glucose between 4-8 AM
        - Caused by cortisol, growth hormone release
        - Management: Increase basal insulin overnight (pump) or take long-acting later

        Somogyi Effect (Rebound Hyperglycemia):
        - High morning glucose following overnight hypoglycemia
        - Counter-regulatory hormones cause glucose spike
        - Management: Reduce overnight basal, check 3 AM glucose

        Distinguish by checking 3 AM glucose:
        - Dawn phenomenon: Normal at 3 AM, high at wake
        - Somogyi: Low at 3 AM, high at wake
        """
    },

    # Alcohol
    {
        "id": "alcohol_1",
        "category": "lifestyle",
        "title": "Alcohol and Diabetes",
        "content": """
        Alcohol effects on glucose:
        - Initially may raise glucose (especially beer, sweet drinks)
        - Later can cause severe hypoglycemia (liver busy processing alcohol)
        - Risk of hypoglycemia extends 12-24 hours after drinking

        Guidelines:
        1. Never drink on an empty stomach
        2. Limit to moderate amounts (1 drink women, 2 men per day)
        3. Reduce insulin before bed after drinking
        4. Set alarm to check glucose overnight
        5. Have snack before bed
        6. Wear medical ID
        7. Companions should know glucagon administration
        """
    },

    # DKA Warning Signs
    {
        "id": "dka_1",
        "category": "emergencies",
        "title": "Diabetic Ketoacidosis (DKA) Warning",
        "content": """
        DKA Warning Signs (Medical Emergency):
        - Blood glucose >250 mg/dL
        - Moderate/large ketones
        - Nausea/vomiting
        - Abdominal pain
        - Fruity breath odor
        - Rapid breathing
        - Confusion/altered consciousness

        SEEK IMMEDIATE MEDICAL CARE if these signs present.

        Prevention:
        - Never skip insulin doses
        - Check ketones when glucose >250 mg/dL
        - Have sick day management plan
        - Check pump site regularly for failures
        """
    },
]


class MedicalGuidelinesRAG:
    """
    RAG system for medical guidelines retrieval.

    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    """

    def __init__(
        self,
        persist_directory: str = "./data/vectors",
        collection_name: str = "medical_guidelines",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Diabetes management guidelines"}
        )

        logger.info(f"Initialized RAG with collection: {collection_name}")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def load_guidelines(self, guidelines: Optional[list[dict]] = None):
        """Load medical guidelines into the vector store."""
        if guidelines is None:
            guidelines = DIABETES_GUIDELINES

        # Check if already loaded
        existing_count = self.collection.count()
        if existing_count >= len(guidelines):
            logger.info(f"Guidelines already loaded ({existing_count} documents)")
            return

        logger.info(f"Loading {len(guidelines)} guidelines into vector store")

        ids = []
        documents = []
        metadatas = []

        for guideline in guidelines:
            ids.append(guideline["id"])
            documents.append(f"{guideline['title']}\n\n{guideline['content']}")
            metadatas.append({
                "category": guideline["category"],
                "title": guideline["title"],
            })

        # Generate embeddings
        embeddings = self._embed(documents)

        # Upsert to collection
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Loaded {len(guidelines)} guidelines successfully")

    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for relevant guidelines.

        Args:
            query: Search query
            n_results: Number of results to return
            category_filter: Optional category to filter by

        Returns:
            List of matching guidelines with metadata
        """
        # Generate query embedding
        query_embedding = self._embed([query])[0]

        # Build where filter
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "category": results["metadatas"][0][i].get("category"),
                "title": results["metadatas"][0][i].get("title"),
                "relevance_score": 1 - results["distances"][0][i],  # Convert distance to similarity
            })

        return formatted_results

    def get_context_for_glucose(
        self,
        current_glucose: float,
        predicted_glucose: float,
        recent_events: Optional[dict] = None,
    ) -> str:
        """
        Get relevant context based on glucose values and patient state.

        Args:
            current_glucose: Current glucose reading
            predicted_glucose: Predicted future glucose
            recent_events: Dict with recent meals, insulin, exercise info

        Returns:
            Concatenated relevant guidelines as context string
        """
        queries = []

        # Query based on current state
        if current_glucose < 70:
            queries.append("hypoglycemia treatment low blood sugar")
        elif current_glucose < 54:
            queries.append("severe hypoglycemia emergency treatment")
        elif current_glucose > 250:
            queries.append("high blood sugar hyperglycemia ketones")
        elif current_glucose > 180:
            queries.append("correction dose high glucose")

        # Query based on prediction
        if predicted_glucose < 70:
            queries.append("preventing hypoglycemia low glucose prevention")
        elif predicted_glucose > 180:
            queries.append("preventing hyperglycemia high glucose")

        # Query based on recent events
        if recent_events:
            if recent_events.get("recent_meal"):
                queries.append("carbohydrate counting meal bolus timing")
            if recent_events.get("recent_exercise"):
                queries.append("exercise glucose management insulin adjustment")
            if recent_events.get("is_dawn_window"):
                queries.append("dawn phenomenon morning glucose")

        # Default query for normal range
        if not queries:
            queries.append("diabetes management time in range targets")

        # Search and deduplicate results
        all_results = []
        seen_ids = set()

        for query in queries:
            results = self.search(query, n_results=2)
            for result in results:
                if result["id"] not in seen_ids:
                    seen_ids.add(result["id"])
                    all_results.append(result)

        # Sort by relevance and take top 5
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        top_results = all_results[:5]

        # Format as context string
        context_parts = ["**Relevant Medical Guidelines:**\n"]
        for result in top_results:
            context_parts.append(f"### {result['title']}")
            context_parts.append(result["content"])
            context_parts.append("")

        return "\n".join(context_parts)


def setup_rag(persist_directory: str = "./data/vectors") -> MedicalGuidelinesRAG:
    """Initialize and setup the RAG system with default guidelines."""
    rag = MedicalGuidelinesRAG(persist_directory=persist_directory)
    rag.load_guidelines()
    return rag
