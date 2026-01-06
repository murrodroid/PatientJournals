from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Address(BaseModel):
    street: Optional[str] = Field(
        default=None,
        description="The (historical) street of the address in Copenhagen. Can also be historical addresses, meaning they may no longer exist in Copenhagen. Typically found in the center of the page."
    )
    number: Optional[str] = Field(
        default=None,
        description="Address number of the street. Written to the right of the street."
    )
    floor: Optional[str] = Field(
        default=None,
        description="Written right of the address number, describing the (danish-system) apartment floor. Typically written with smaller font size. It's often a number, sometimes text, sometimes both. Examples: [3, 1ste, St., kælderen]"
    )

class HospitalStay(BaseModel):
    admissionDate: Optional[datetime] = Field(
        default=None,
        description="Written after 'Indl', and the upper-most date on the page. May feature additional text indicating time of day, otherwise assume hour 0. If year isn't written, year is assumed as 1896."
    )
    releaseDate: Optional[datetime] = Field(
        default=None,
        description="Written after 'Udskr', beneath admissionDate. May feature additional text indicating time of day, otherwise assume hour 0. If year isn't written, year is assumed as 1896."
    )
    stayLength: Optional[str] = Field(
        default=None,
        description="Written beneath the release date, 'releaseDate'. Commonly a number describing days, but sometimes also indication of hours spent. Examples: [2 Dage (11 Timer), 5.5 Timer, 9 Dage, 3]"
    )

class PatientAge(BaseModel):
    age_num: Optional[float] = Field(
        default=None,
        description="The age of the patient. Written next to 'Aar'. In some cases, there are multiple guesses of age, ie '2 (2.5?)'. In the case of additional notation, only use the initial age mentioned (in this case, 2)."
    )
    age_note: Optional[str] = Field(
        default=None,
        description="All additional text included after the age. Examples: [Aar, (2.5?)]"
    )

class Journal(BaseModel):
    index: int = Field(
        description="Index value of the patient journal. Typically written above the patient's name."
    )
    name: str = Field(
        description="The name of the patient."
    )
    ward: Optional[str] = Field(
        default=None,
        description="Letter+number combination found in the top-left corner of the page. Numbers may sometimes be written with roman numerals."
    )
    is_dead: bool = Field(
        default=False,
        description="Determines whether the patient is dead. If true, a black cross is drawn on the page, typically at the top. Sometimes it looks like a plus, '+'. If the cross does not appear, it is False."
    )

    age: Optional[PatientAge] = Field(
        default=None,
        description="The age of the patient."
    )
    address: Optional[Address] = Field(
        default=None,
        description="The patient's residence information, typically found in the center of the page."
    )
    stay: Optional[HospitalStay] = Field(
        default=None,
        description="Information regarding the hospital admission and release."
    )
