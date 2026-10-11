from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class SubscriptionType(str, Enum): 
    BASIC = "Basic"; STANDARD = "Standard"; PREMIUM = "Premium"

class ContractLength(str, Enum): 
    MONTHLY = "Monthly"; QUARTERLY = "Quarterly"; ANNUAL = "Annual"

class CustomerData(BaseModel):
    CustomerID: int = Field(..., example=101)
    Age: int = Field(..., example=30)
    Gender: str = Field(..., example="Female")
    Tenure: int = Field(..., example=12)
    Usage_Frequency: int = Field(..., alias="Usage Frequency", example=15)
    Support_Calls: int = Field(..., alias="Support Calls", example=2)
    Payment_Delay: int = Field(..., alias="Payment Delay", example=5)
    Subscription_Type: SubscriptionType = Field(..., alias="Subscription Type", example=SubscriptionType.STANDARD)
    Contract_Length: ContractLength = Field(..., alias="Contract Length", example=ContractLength.ANNUAL)
    Total_Spend: float = Field(..., alias="Total Spend", example=450.50)
    Last_Interaction: int = Field(..., alias="Last Interaction", example=3)
    class Config: populate_by_name = True

# other schemas
class UserAuth(BaseModel): username: str; password: str
class ReqSchema(BaseModel): data: CustomerData
class Batch(BaseModel): data: List[CustomerData]
class Label(BaseModel): request_id: str; churn: int