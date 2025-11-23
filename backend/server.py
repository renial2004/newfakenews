from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from textblob import TextBlob
import nltk
from urllib.parse import urlparse
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Initialize ML model and vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
model = None

# Training data for fake news detection
training_texts = [
    # Fake news examples
    "BREAKING: Scientists discover miracle cure that doctors don't want you to know about!",
    "SHOCKING: Celebrity caught in unbelievable scandal that will blow your mind!",
    "You won't believe what happened next! Click here for amazing results!",
    "This one weird trick will change your life forever! Doctors hate him!",
    "URGENT: Share this before it gets deleted! The truth they're hiding from you!",
    "Experts say this is the most dangerous thing ever discovered!",
    "Breaking news: Politician admits to unthinkable crime in secret recording!",
    "Studies show 100% of people who try this get rich overnight!",
    # Real news examples
    "The Federal Reserve announced a 0.25% interest rate adjustment following their quarterly meeting.",
    "Researchers at Stanford University published findings on climate change patterns in peer-reviewed journal.",
    "The stock market experienced moderate volatility today amid economic data releases.",
    "Local government approved budget increase for public infrastructure improvements.",
    "Health officials recommend annual checkups and preventive care measures.",
    "Technology company reports quarterly earnings in line with analyst expectations.",
    "Scientists continue research into renewable energy solutions with promising results.",
    "Education department releases annual report on student achievement metrics.",
]

training_labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # 0=fake, 1=real

# Initialize model on startup
def initialize_model():
    global model, vectorizer
    X = vectorizer.fit_transform(training_texts)
    model = LogisticRegression(random_state=42)
    model.fit(X, training_labels)
    logging.info("ML model initialized successfully")

# Models
class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    
class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    classification: str  # "Verified", "Suspicious", "Fake"
    confidence: float
    category: str
    sentiment: str
    sensationalism_score: float
    source_credibility: float
    explanation: str
    summary: str
    linguistic_patterns: List[str]
    metadata: dict
    original_text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    crisis_flag: bool = False
    crisis_reason: Optional[str] = None

class HistoryItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    classification: str
    confidence: float
    category: str
    summary: str
    timestamp: str
    saved_status: Optional[str] = None

# Helper functions
def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
    text = text.lower().strip()
    return text

def analyze_sensationalism(text: str) -> float:
    """Detect sensationalist language"""
    sensational_words = [
        'shocking', 'unbelievable', 'amazing', 'incredible', 'miracle',
        'secret', 'hidden', 'truth', 'exposed', 'revealed', 'breaking',
        'urgent', 'must see', 'won\'t believe', 'mind-blowing', 'you won\'t',
        'doctors hate', 'they don\'t want you to know'
    ]
    
    text_lower = text.lower()
    count = sum(1 for word in sensational_words if word in text_lower)
    
    # Check for excessive punctuation
    exclamation_count = text.count('!')
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    score = min((count * 15 + exclamation_count * 5 + caps_ratio * 50) / 100, 1.0)
    return round(score, 2)

def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.3:
        return "Positive"
    elif polarity < -0.3:
        return "Negative"
    else:
        return "Neutral"

def classify_category(text: str) -> str:
    """Classify article category"""
    categories = {
        'Politics': ['government', 'election', 'president', 'politics', 'politician', 'vote', 'congress', 'senate'],
        'Health': ['health', 'medical', 'doctor', 'hospital', 'disease', 'treatment', 'medicine', 'cure'],
        'Entertainment': ['celebrity', 'movie', 'music', 'actor', 'actress', 'film', 'entertainment'],
        'Finance': ['stock', 'market', 'economy', 'financial', 'money', 'investment', 'bank', 'business'],
        'Technology': ['technology', 'tech', 'computer', 'software', 'digital', 'internet', 'app'],
        'Science': ['science', 'research', 'study', 'scientist', 'discovery', 'experiment'],
    }
    
    text_lower = text.lower()
    category_scores = {}
    
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        category_scores[category] = score
    
    if max(category_scores.values()) > 0:
        return max(category_scores, key=category_scores.get)
    return "General"

def calculate_source_credibility(url: Optional[str]) -> float:
    """Calculate source credibility score"""
    if not url:
        return 0.5
    
    trusted_domains = ['reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com', 'washingtonpost.com']
    suspicious_tlds = ['.xyz', '.tk', '.ml', '.ga']
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        if any(trusted in domain for trusted in trusted_domains):
            return 0.9
        
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            return 0.2
        
        return 0.5
    except:
        return 0.5

def detect_linguistic_patterns(text: str) -> List[str]:
    """Detect suspicious linguistic patterns"""
    patterns = []
    
    if text.count('!') > 3:
        patterns.append("Excessive exclamation marks")
    
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.3:
        patterns.append("Excessive capitalization")
    
    if any(phrase in text.lower() for phrase in ['click here', 'click now', 'limited time']):
        patterns.append("Clickbait phrases detected")
    
    if re.search(r'\d+%', text):
        patterns.append("Statistical claims without source")
    
    sensational = ['shocking', 'unbelievable', 'miracle', 'secret']
    if sum(1 for word in sensational if word in text.lower()) > 2:
        patterns.append("Sensationalist language")
    
    return patterns if patterns else ["Standard journalistic tone"]

def generate_summary(text: str, max_sentences: int = 2) -> str:
    """Generate text summary"""
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
        
        # Simple extractive summarization
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        word_freq = {}
        
        for word in words:
            if word.isalnum() and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
        
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
        return ' '.join(summary_sentences)
    except:
        return text[:200] + "..."

def detect_crisis_content(text: str) -> tuple[bool, Optional[str]]:
    """Detect potentially harmful content"""
    hate_keywords = ['hate', 'violence', 'attack', 'threat', 'kill', 'destroy']
    political_hate = ['enemy', 'traitor', 'destroy them', 'fight back']
    
    text_lower = text.lower()
    
    hate_count = sum(1 for word in hate_keywords if word in text_lower)
    political_count = sum(1 for phrase in political_hate if phrase in text_lower)
    
    if hate_count >= 2 or political_count >= 1:
        return True, "Content contains potentially harmful political rhetoric or hate speech"
    
    return False, None

@api_router.post("/analyze", response_model=AnalysisResult)
async def analyze_content(request: AnalysisRequest):
    """Analyze text or URL for fake news"""
    try:
        if not request.text and not request.url:
            raise HTTPException(status_code=400, detail="Either text or URL must be provided")
        
        # Get text content
        text = request.text or "Sample news article from URL"
        
        # Clean and analyze
        cleaned_text = clean_text(text)
        
        # ML prediction
        X = vectorizer.transform([cleaned_text])
        prediction = model.predict(X)[0]
        confidence_scores = model.predict_proba(X)[0]
        confidence = float(confidence_scores[prediction])
        
        # Additional analysis
        sensationalism = analyze_sensationalism(text)
        sentiment = analyze_sentiment(text)
        category = classify_category(text)
        source_credibility = calculate_source_credibility(request.url)
        linguistic_patterns = detect_linguistic_patterns(text)
        summary = generate_summary(text)
        crisis_flag, crisis_reason = detect_crisis_content(text)
        
        # Determine classification
        if prediction == 1 and confidence > 0.7 and sensationalism < 0.3:
            classification = "Verified"
        elif prediction == 0 or sensationalism > 0.5:
            classification = "Fake"
        else:
            classification = "Suspicious"
        
        # Adjust confidence based on multiple factors
        adjusted_confidence = (confidence * 0.5 + (1 - sensationalism) * 0.3 + source_credibility * 0.2)
        
        # Generate explanation
        explanation_parts = []
        if sensationalism > 0.4:
            explanation_parts.append(f"High sensationalism score ({sensationalism:.0%})")
        if source_credibility < 0.4:
            explanation_parts.append("Low source credibility")
        if linguistic_patterns and linguistic_patterns[0] != "Standard journalistic tone":
            explanation_parts.append(f"Detected: {', '.join(linguistic_patterns[:2])}")
        
        explanation = "; ".join(explanation_parts) if explanation_parts else "Standard news format detected"
        
        result = AnalysisResult(
            classification=classification,
            confidence=round(adjusted_confidence, 2),
            category=category,
            sentiment=sentiment,
            sensationalism_score=sensationalism,
            source_credibility=source_credibility,
            explanation=explanation,
            summary=summary,
            linguistic_patterns=linguistic_patterns,
            metadata={
                "text_length": len(text),
                "word_count": len(text.split()),
                "has_url": bool(request.url)
            },
            original_text=text[:500],
            crisis_flag=crisis_flag,
            crisis_reason=crisis_reason
        )
        
        # Save to database
        doc = result.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.analyses.insert_one(doc)
        
        return result
        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/history", response_model=List[HistoryItem])
async def get_history():
    """Get analysis history"""
    try:
        history = await db.analyses.find({}, {"_id": 0}).sort("timestamp", -1).limit(50).to_list(50)
        
        items = []
        for item in history:
            items.append(HistoryItem(
                id=item.get('id'),
                classification=item.get('classification'),
                confidence=item.get('confidence'),
                category=item.get('category'),
                summary=item.get('summary', '')[:100],
                timestamp=item.get('timestamp'),
                saved_status=item.get('saved_status')
            ))
        
        return items
    except Exception as e:
        logging.error(f"History error: {str(e)}")
        return []

@api_router.post("/save-status/{analysis_id}")
async def save_status(analysis_id: str, status: dict):
    """Update saved status for an analysis"""
    try:
        result = await db.analyses.update_one(
            {"id": analysis_id},
            {"$set": {"saved_status": status.get('status')}}
        )
        return {"success": result.modified_count > 0}
    except Exception as e:
        logging.error(f"Save status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/stats")
async def get_stats():
    """Get statistics for dashboard"""
    try:
        total = await db.analyses.count_documents({})
        fake_count = await db.analyses.count_documents({"classification": "Fake"})
        verified_count = await db.analyses.count_documents({"classification": "Verified"})
        suspicious_count = await db.analyses.count_documents({"classification": "Suspicious"})
        
        # Category distribution
        categories = await db.analyses.aggregate([
            {"$group": {"_id": "$category", "count": {"$sum": 1}}}
        ]).to_list(100)
        
        return {
            "total_analyses": total,
            "fake_count": fake_count,
            "verified_count": verified_count,
            "suspicious_count": suspicious_count,
            "categories": {cat['_id']: cat['count'] for cat in categories}
        }
    except Exception as e:
        logging.error(f"Stats error: {str(e)}")
        return {}

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    initialize_model()
    logger.info("Application started successfully")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()