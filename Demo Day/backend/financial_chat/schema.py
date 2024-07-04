from langchain.pydantic_v1 import BaseModel, Field
from enum import Enum
from typing import Literal, List


class FilingItem(Enum):
    ITEM_1 = "Item 1"
    ITEM_1A = "Item 1A"
    ITEM_1B = "Item 1B"
    ITEM_2 = "Item 2"
    ITEM_3 = "Item 3"
    ITEM_4 = "Item 4"
    ITEM_5 = "Item 5"
    ITEM_6 = "Item 6"
    ITEM_7 = "Item 7"
    ITEM_7A = "Item 7A"
    ITEM_8 = "Item 8"
    ITEM_9 = "Item 9"
    ITEM_9A = "Item 9A"
    ITEM_9B = "Item 9B"
    ITEM_10 = "Item 10"
    ITEM_11 = "Item 11"
    ITEM_12 = "Item 12"
    ITEM_13 = "Item 13"
    ITEM_14 = "Item 14"
    ITEM_15 = "Item 15"
    ITEM_16 = "Item 16"


class ReportParams(BaseModel):
    """
    The parameters to get a company's filing information for a specific year and form.
    """

    ticker: str = Field(
        ..., description="The ticker of the company.", max_length=4, min_length=1
    )
    year: int = Field(..., description="The year of the filing.", ge=1990, le=2023)
    form: Literal["10-K", "10-Q"] = Field(..., description="The form of the filing.")
    item_names: List[FilingItem] = Field(
        None, description="The items to get from the filing."
    )
