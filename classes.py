from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Journal(BaseModel):
    name: str       = Field(description="The name of the patient.")
    age: float      = Field(description="The age of the patient. Written next to 'Aar'. In some cases, there are multiple guesses of age, ie '2 (2.5?)'. In the case of additional notation, only use the initial age mentioned (in this case, 2).")
    age_note: str   = Field(description="All additional data included after the age. Examples: [Aar, (2.5?)]")
    index: int      = Field(description="Index value of the patient journal. Typically written above the patient's name.")
    ward: str       = Field(description="Letter+number combination found in the top-left corner of the page. Numbers may sometimes be written with roman numerals.")
    addressStreet: str = Field(description="The (historical) street of the address in Copenhagen. Can also be historical adresses, meaning they may no longer exist in Copenhagen. Typically found in the center of the page.")
    addressNumber: int = Field(description="Address number on the street. Strictly a single integer, written next to the street.")
    addressApartment: str = Field(description="Written left of the address number, describing the (danish-system) apartment floor. Typically written with smaller font size. It's often a number, sometimes text, sometimes both. Examples: [3, 1ste, St., kælderen]")
    dead: bool = Field(description="Determines whether the patient is dead. If true, a black cross is drawn on the page, typically at the top. Sometimes it looks like a plus, '+'. If the cross does not appear, it is False.")
    admissionDate: datetime = Field(description="Written after “Indl”, and the upper-most date on the page. May feature additional text indicating time of day, otherwise assume hour 0. If year isn't written, remember to add year 1890.")
    releaseDate: datetime = Field(description="Written after 'Udskr', beneath admissionDate. May feature additional text indicating time of day, otherwise assume hour 0. If year isn't written, remember to add year 1890.")
    stayLength: str = Field(description="Written beneath the release date. Commonly a number describing days, but sometimes also indication of hours spent. Examples: [2 Dage (11 Timer), 5.5 Timer, 9 Dage, 3]")