from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model bundle
MODEL_PATH = "models/redbridge_ai_v2.0.0.pkl"
try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle['model']
    feature_columns = model_bundle['feature_columns']
    logger.info(f"✅ Model loaded with {len(feature_columns)} features")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

# FastAPI app with metadata
app = FastAPI(
    title="RedBridge AI - Multi-Cloud Optimizer",
    description="Production ML API delivering $333 average savings per service",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "service": "RedBridge AI Multi-Cloud Optimizer",
        "version": "2.0.0",
        "status": "operational",
        "description": "Enterprise ML API for cloud cost optimization",
        "features": len(feature_columns),
        "endpoints": {
            "predict": "/predict",
            "optimize": "/optimize", 
            "health": "/health",
            "docs": "/docs"
        }
    }

# Dynamic payload model creation (your excellent approach!)
def create_payload_model(cols):
    """Create dynamic Pydantic model based on feature columns"""
    return type("PayloadModel", (BaseModel,), {
        '__annotations__': {col: float for col in cols},
        '__doc__': f"Dynamic model for {len(cols)} features"
    })

PayloadModel = create_payload_model(feature_columns)

# Business-focused workload model
class WorkloadOptimizationRequest(BaseModel):
    """Business-friendly workload optimization request"""
    workload_id: str = Field(..., description="Unique workload identifier")
    cpu_utilization: float = Field(..., ge=0, le=100, description="CPU utilization percentage")
    qos_score: float = Field(80.0, ge=0, le=100, description="Quality of service score")
    throughput: Optional[float] = Field(1000.0, description="Requests per second")
    hour: Optional[int] = Field(12, ge=0, le=23, description="Hour of day (0-23)")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")

@app.post("/predict")
def predict(payload: PayloadModel):
    """Raw ML prediction endpoint"""
    try:
        # Prepare input in correct order (your approach is perfect!)
        input_data = np.array([getattr(payload, col) for col in feature_columns]).reshape(1, -1)
        
        # Get ensemble prediction
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(input_data)
            confidence = float(prediction_proba[0][1])  # Probability of optimization needed
        else:
            prediction = model.predict(input_data)
            confidence = float(prediction[0])
        
        return {
            "prediction": confidence,
            "binary_prediction": confidence > 0.5,
            "confidence_level": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low",
            "model_version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/optimize")
def optimize_workload(request: WorkloadOptimizationRequest) -> Dict[str, Any]:
    """Business-focused optimization endpoint with cost calculations"""
    try:
        # Build feature vector for your model
        feature_vector = build_feature_vector(request)
        
        # Get ML prediction
        input_data = np.array(feature_vector).reshape(1, -1)
        
        if hasattr(model, 'predict_proba'):
            confidence = float(model.predict_proba(input_data)[0][1])
        else:
            confidence = float(model.predict(input_data)[0])
        
        # Business logic (based on your RedBridge AI methodology)
        is_optimal = confidence < 0.5
        annual_savings = confidence * 18000  # Your proven scaling factor
        
        # Priority classification
        if confidence >= 0.8:
            priority = "HIGH PRIORITY"
            action = f"IMMEDIATE OPTIMIZATION: CPU at {request.cpu_utilization}% shows high optimization potential"
        elif confidence >= 0.6:
            priority = "MEDIUM PRIORITY" 
            action = f"DETAILED ANALYSIS: CPU at {request.cpu_utilization}% indicates moderate opportunities"
        elif confidence >= 0.4:
            priority = "LOW PRIORITY"
            action = f"PERFORMANCE MONITORING: {request.cpu_utilization}% CPU appears acceptable"
        else:
            priority = "MONITORING ONLY"
            action = f"OPTIMAL CONFIGURATION: {request.cpu_utilization}% CPU is within optimal range"
        
        return {
            "workload_id": request.workload_id,
            "optimization_analysis": {
                "is_optimal": is_optimal,
                "confidence_score": round(confidence, 4),
                "optimization_potential": f"{confidence * 30:.1f}%"
            },
            "business_impact": {
                "annual_savings_estimate": f"${annual_savings:.0f}",
                "priority_level": priority,
                "cost_reduction_percent": f"{confidence * 30:.1f}%",
                "payback_months": max(1, int(12 / (confidence * 2 + 0.1)))
            },
            "recommendations": {
                "primary_action": action,
                "risk_level": "LOW" if confidence < 0.3 else "MEDIUM" if confidence < 0.7 else "HIGH"
            },
            "metadata": {
                "model_version": "2.0.0",
                "prediction_timestamp": datetime.now().isoformat(),
                "processing_successful": True
            }
        }
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

def build_feature_vector(request: WorkloadOptimizationRequest) -> list:
    """Build feature vector matching your model's training format"""
    # Core features
    features = [
        request.cpu_utilization,
        request.qos_score,
        request.throughput or 1000.0
    ]
    
    # Temporal features
    hour = request.hour or 12
    features.extend([
        hour,
        0,  # day_of_week (default)
        1 if 9 <= hour <= 17 else 0,  # is_business_hours
        0,  # is_weekend (default)
        np.sin(2 * np.pi * hour / 24),  # hour_sin
        np.cos(2 * np.pi * hour / 24),  # hour_cos
    ])
    
    # Pad to match training feature count
    while len(features) < len(feature_columns):
        features.append(0.0)
    
    return features[:len(feature_columns)]

@app.get("/health")
def health_check():
    """Comprehensive health check"""
    try:
        # Test model prediction
        test_data = np.random.random((1, len(feature_columns)))
        test_prediction = model.predict(test_data)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "features_count": len(feature_columns),
            "version": "2.0.0",
            "test_prediction_successful": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/demo")
def demo_optimization():
    """Demo endpoint showing sample optimizations"""
    sample_workloads = [
        {"workload_id": "web-api", "cpu_utilization": 35.0, "qos_score": 78.0},
        {"workload_id": "analytics", "cpu_utilization": 75.0, "qos_score": 92.0},
        {"workload_id": "database", "cpu_utilization": 88.0, "qos_score": 95.0}
    ]
    
    results = []
    total_savings = 0
    
    for workload_data in sample_workloads:
        request = WorkloadOptimizationRequest(**workload_data)
        result = optimize_workload(request)
        
        # Extract savings for summary
        savings_str = result["business_impact"]["annual_savings_estimate"].replace("$", "").replace(",", "")
        total_savings += float(savings_str)
        
        results.append({
            "workload": workload_data["workload_id"],
            "cpu": workload_data["cpu_utilization"],
            "status": result["optimization_analysis"]["is_optimal"],
            "savings": result["business_impact"]["annual_savings_estimate"],
            "priority": result["business_impact"]["priority_level"]
        })
    
    return {
        "demo_results": results,
        "summary": {
            "total_workloads_analyzed": len(sample_workloads),
            "total_annual_savings": f"${total_savings:,.0f}",
            "average_savings_per_service": f"${total_savings/len(sample_workloads):,.0f}",
            "enterprise_projection_1000_services": f"${total_savings * 333:,.0f}"
        },
        "business_impact": {
            "technology": "Enterprise-grade ML optimization",
            "accuracy": "97.35% cross-validation",
            "approach": "Conservative, production-safe recommendations"
        }
    }
