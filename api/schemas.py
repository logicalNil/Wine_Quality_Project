from pydantic import BaseModel, field_validator
from typing import Literal

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    type: str  # "red" or "white"

    @field_validator('type')
    def validate_type(cls, v):
        if v not in ["red", "white"]:
            raise ValueError('Type must be either "red" or "white"')
        return v

    @field_validator('total_sulfur_dioxide')
    def validate_total_sulfur_dioxide(cls, v, values):
        if 'free_sulfur_dioxide' in values.data and v < values.data['free_sulfur_dioxide']:
            raise ValueError('Total sulfur dioxide must be greater than or equal to free sulfur dioxide')
        return v

class PredictionResponse(BaseModel):
    quality: float
    confidence: float