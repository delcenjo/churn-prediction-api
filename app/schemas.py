from enum import Enum

from pydantic import BaseModel, Field


class Contract(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class InternetService(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class CustomerFeatures(BaseModel):
    tenure: int = Field(ge=0, le=120, description="Number of months as a customer")
    monthly_charges: float = Field(ge=0, description="Current monthly charge")
    total_charges: float = Field(ge=0, description="Total amount charged to date")
    contract: Contract
    internet_service: InternetService

    model_config = {
        "use_enum_values": True,
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 2,
                    "monthly_charges": 89.9,
                    "total_charges": 179.8,
                    "contract": "Month-to-month",
                    "internet_service": "Fiber optic",
                }
            ]
        },
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn: bool
    threshold: float
