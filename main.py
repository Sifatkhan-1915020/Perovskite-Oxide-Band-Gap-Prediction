from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
from pymatgen.core import Composition
import joblib
import numpy as np

app = FastAPI(title="Perovskite Dual Predictor")

# ==========================================
# 1. Load Pre-trained Models & Scaler
# ==========================================
try:
    regressor_model = joblib.load('perovskite_regressor.pkl')
    classifier_model = joblib.load('perovskite_classifier.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    print("Models and Scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Missing .pkl files. Please run both training scripts first.")

# ==========================================
# 2. Setup the Embedding Layer
# ==========================================
embed_dim = 8
embedding_layer = nn.Embedding(num_embeddings=119, embedding_dim=embed_dim)

def is_perovskite(formula: str) -> bool:
    """
    Validates if the formula fits the basic Oxide Perovskite stoichiometry (ABO3).
    Checks if Oxygen makes up roughly 60% (3/5) of the atomic composition.
    """
    try:
        comp = Composition(formula.strip())
        fractions = comp.fractional_composition
        
        o_fraction = 0.0
        for element, fraction in fractions.items():
            if element.symbol == 'O':
                o_fraction = fraction
                break
                
        # If there is no Oxygen, it's not a metal oxide perovskite
        if o_fraction == 0.0:
            return False
            
        # Check if Oxygen fraction is around 0.6 (allowing a 5% margin for defects/vacancies)
        if abs(o_fraction - 0.6) > 0.05:
            return False
            
        return True
    except Exception:
        return False

def get_formula_embedding(formula: str):
    try:
        comp = Composition(formula.strip())
        atom_ids = []
        atom_fractions = []
        for element, fraction in comp.fractional_composition.items():
            atom_ids.append(element.Z)
            atom_fractions.append(fraction)
            
        ids_tensor = torch.tensor(atom_ids)
        fractions_tensor = torch.tensor(atom_fractions, dtype=torch.float32).unsqueeze(-1)
        
        with torch.no_grad():
            embeds = embedding_layer(ids_tensor)
            weighted_embeds = embeds * fractions_tensor
            formula_vector = torch.sum(weighted_embeds, dim=0)
            
        return [round(x, 4) for x in formula_vector.tolist()]
    except Exception as e:
        raise ValueError(f"Invalid chemical formula: {str(e)}")

# ==========================================
# 3. Define API Data Models
# ==========================================
class PredictionRequest(BaseModel):
    formula: str
    volume: float
    density: float
    formation_energy: float
    energy_above_hull: float

# ==========================================
# 4. Define Endpoints
# ==========================================
@app.post("/api/predict")
def predict_band_gap(data: PredictionRequest):
    try:
        # Step 1: Validate if it's an oxide perovskite
        if not is_perovskite(data.formula):
            raise HTTPException(
                status_code=400, 
                detail="Input rejected: Formula does not match standard Oxide Perovskite stoichiometry (ABO3)."
            )

        # Step 2: Embed the formula
        embed_vector = get_formula_embedding(data.formula)
        
        # Step 3: Combine and scale features
        features = [
            data.volume,
            data.density,
            data.formation_energy,
            data.energy_above_hull
        ] + embed_vector
        
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Step 4: Dual Predictions
        reg_pred = regressor_model.predict(features_scaled)[0]
        clf_pred = classifier_model.predict(features_scaled)[0]
        
        class_label = "Wide Band Gap (> 2.0 eV)" if clf_pred == 1 else "Narrow Band Gap (<= 2.0 eV)"
        
        return {
            "status": "success", 
            "formula": data.formula,
            "predicted_ev": round(float(reg_pred), 4),
            "band_gap_class": class_label,
            "raw_class": int(clf_pred)
        }
        
    except HTTPException:
        # Re-raise FastAPI HTTP exceptions so they are sent correctly to the frontend
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("index.html", "r") as f:
        return f.read()
