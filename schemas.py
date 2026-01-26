from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

_SCHEMA_VERSION = 1.0

class Address(BaseModel):
    street: str = Field(
        description="The (historical) street of the address. Can also be historical addresses, meaning they may no longer exist. Can also be the name of an institution. Written as the first, left-most, part of the address."
    )
    number: str = Field(
        description="Address number of the street. Written to the right of the street name. Often contains an integer and sometimes a letter."
    )
    apt: Optional[str] = Field(
        default=None,
        description="Written right of the address number, describing the (danish-system) apartment floor. Typically written with smaller font size. It's often a number, sometimes text, sometimes both. Examples: [3, 1ste, St., kælderen]"  
    )

class Age(BaseModel):
    num: float = Field(
        description="The age of the patient. After name. In some cases, there are multiple guesses of age, ie '2 (2.5?)'. In the case of additional notation, only use the initial age mentioned (in this case, 2). Can also include a division like 11/12, which then indicates a float-value. For 11/12 it would be 0.92."
    )
    unit: str = Field(
        default="Aar",
        description="The unit of the age, most often “Aar” or “år” but can also be an indication of months, “mnd” or days “dage”. Written next to the numerical value of the age."
    )
    note: Optional[str] = Field(
        default=None,
        description="All additional text included after the age. Mostly used to capture uncertainty about the age. Examples: [Aar, (2.5?)]"
    )

class Patient(BaseModel):
    number: Optional[str] = Field(
        default=None,
        description="The patient number in the center of the page, above the patient name. Most often an integer, but can in rare cases include a letter as well. "
    )
    name: str = Field(
        description="The name of the patient. Written underneath the patient number (if present) and above the occupation."
    )
    household_position: str = Field(
        description="Underneath the patient name. Can be an occupation, relational, or marital status of the patient. This information can be combined in various ways, write the entire string."
    )
    age: Age = Field(
        description="Age is written after name, either on the same line, or the following."
    )
    address: Address = Field(
        description="The address is written after household_position and as the last part of the central section of the page."
    )

class Bottom(BaseModel):
    doctor_name: str = Field(
        description="Name of the doctor/physician conducting the journal. Usually as the last line in the bottom right of the page, but may also be just above the bottom diagnosis."
    )
    diagnosis: str = Field(
        description="A diagnosis found in the bottom right corner of the page, usually above the doctor’s name, but can be the last row of the page as well. "
    )

class Sektion(BaseModel):
    number: int = Field(
        description="The dissection number, written at the top after “s.d.” or “sektionsdiagnose” or similar. Usually underlined and sometimes within parentheses. "
    )
    diagnoses: List[str] = Field(
        description="Consists of medical diagnoses or symptoms. Usually contains multiple items, each item defined by a new row. An item often includes a ditto dash, indicating that it is the same word as the one(s) immediately above it."
    )

class HospitalStay(BaseModel):
    admission_date: date = Field(
        description="Written after 'Indl' or 'indskrevet' or similar indication of admission, and the upper-most date on the page. The year must be between 1879 and 1910."
    )
    release_date: date = Field(
        description="Written after 'Udskr', 'Udskrevet' or similar indication of discharge, beneath admissionDate. May feature additional text indicating time of day. This is not transcribed here but in “note”. The year must be between 1879 and 1910."
    )
    stay_length: str = Field(
        description="Written beneath the release date. Commonly a number describing days, but sometimes also indication of hours (or very rarely, minutes) spent. Examples: [2 Dage (11 Timer), 5.5 Timer, 9 Dage, 3]"
    )
    ward: str = Field(
        description="Often a letter+number combination but can also be a word. Found in the top-left corner of the page. Numbers may sometimes be written with roman numerals. Examples include 'Observation' or 'Officiant', 'tent' or simply 'C1' or 'CI'."
    )
    note: Optional[str] = Field(
        default=None,
        description="Note for additional information after the length of stay, as, in rare cases, there may be additional text indicating time of day."
    )

class Diagnoses(BaseModel):
    top: List[str] = Field(
        description="Usually found in the top right of the page and consists of medical diagnoses and symptoms. Usually it contains multiple items, each item defined by a new row. An item often includes a ditto dash, indicating that it is a continuation with the same word(s) as the one(s) immediately above it."
    )
    bottom: Bottom = Field(
        description="Diagnosis and doctors name found in the bottom right corner of the page. "
    )
    sektion: Optional[Sektion] = Field(
        default=None,
        description="Only if is_dead is True. Usually found in the bottom half of the journal, below hospital_stay information. First row always contains an indication like: “s.d.” or “sektiondiagnosis”, “sdn”, or similar, followed by a number. "
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severity of the patient. Usually on the left side of the page. Often a word accompanied by '<' or '>' sign. Examples: [<middel, lidt, middelstærk, middelsvær, mild, svær]"
    )

class Serum(BaseModel):
    given: bool = Field(
        description="In the top left, sometimes the test “serum” or similar is found. Value is True if there is any information about serum given."
    )
    doses: Optional[str] = Field(
        default=None,
        description="Indicates the doses of serum given. Only if given is True. Usually an integer, or a list of integers seperated by “+”. Can contain roman numerals as well; often underneath the integers."
    )
    type: Optional[str] = Field(
        description="Type of serum given, often written after 'Serum' (or similar), usually indicating a country name, 'Frs', 'germ', 'dan' for french, german, danish, or similar. These can be followed by a roman numeral. "
    )


class Journal(BaseModel):
    is_dead: bool = Field(
        description="Determines whether the patient is dead. If true, a black cross is drawn on the page, typically at the top. Sometimes it looks like a plus, '+'. If the cross does not appear, it is False."
    )
    is_fk: bool = Field(
        description="In many cases, the page contains “F.K.” or “F K” or “FK”. If this symbol/text appears, the variable is True."
    )
    patient: Patient = Field(
        description="Patient related information like, name, age, address, position in the household and patient number. Generally located around the center of the page. The information is written in late 19th and early 20th century Danish."
    )
    hospital_stay: HospitalStay = Field(
        description="Contains information about the patient's stay at the hospital, like the ward, the admission and discharge dates, and the length of stay. Most information is found underneath the patient information, but ward can be found in the top left corner. Also contains severity information, usually found on the left side of the page and indicating the general severity of the patient. The information is written in late 19th and early 20th century Danish."
    )
    diagnoses: Diagnoses = Field(
        description="Contains all information related to the diagnoses. Contains mainly medical information and terminology. It is divided into several parts: top diagnoses written in the top right comer of the page; bottom diagnoses written in the bottom right comer—which also include the doctor's name; and section diagnoses which are generally found in the (bottom) left part of the page.  The information is written in late 19th and early 20th century Danish."
    )
    serum: Serum = Field(
        description="Contains information on whether the patient was treated with anti-diphtheria serum, and if so, the type of serum and dosis information. Not all cases have this information. If it is found, it is in the top left corner of the journal, underneath or beside the ward information. Often written diagonally."
    )

