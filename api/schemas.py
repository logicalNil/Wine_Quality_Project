from pydantic import BaseModel, Field, validator
from typing import Literal
import numpy as np


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., gt=0, le=15, description="Fixed acidity (tartaric acid) in g/dm³")
    volatile_acidity: float = Field(..., gt=0, le=2, description="Volatile acidity (acetic acid) in g/dm³")
    citric_acid: float = Field(..., ge=0, le=1, description="Citric acid in g/dm³")
    residual_sugar: float = Field(..., ge=0, le=20, description="Residual sugar in g/dm³")
    chlorides: float = Field(..., gt=0, le=0.2, description="Sodium chloride in g/dm³")
    free_sulfur_dioxide: float = Field(..., ge=0, le=100, description="Free sulfur dioxide in mg/dm³")
    total_sulfur_dioxide: float = Field(..., ge=0, le=300, description="Total sulfur dioxide in mg/dm³")
    density: float = Field(..., gt=0.98, lt=1.1, description="Density in g/cm³")
    pH: float = Field(..., gt=0, le=5, description="pH value")
    sulphates: float = Field(..., gt=0, le=2, description="Potassium sulphate in g/dm³")
    alcohol: float = Field(..., gt=0, le=15, description="Alcohol by volume (%)")
    type: Literal["red", "white"] = Field(..., description="Wine type")

    @validator('type')
    def convert_type(cls, v):
        return 0 if v == "red" else 1

    @validator('total_sulfur_dioxide')
    def validate_total_sulfur_dioxide(cls, v, values):
        if 'free_sulfur_dioxide' in values and v < values['free_sulfur_dioxide']:
            raise ValueError('Total sulfur dioxide must be greater than or equal to free sulfur dioxide')
        return v


class PredictionResponse(BaseModel):
    quality_class: int
    probabilities: list
    class_labels: dict