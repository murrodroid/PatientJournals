from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal
from datetime import date

_SCHEMA_VERSION = 2.0

WARD_CLASSES = Literal["O", "Off.", "Obs.", "A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2", "E1", "E2", "F1", "F2", "G1", "G2", "G3", "G4", "H", "J", "K1", "K2", "Barak", "Telt A", "Telt B","Telt C","Telt D","Telt E","Telt F","Telt G","Telt L", "Telt 1", "Telt 2", "Telt 3", "Telt 4", "Telt 5", "Telt 6", "Telt 7", "Telt 8", "Telt 9", "Telt 10", "Telt 11"]
DB_CLASSES = Literal["+DB","-DB","+DB?","-DB?","DB in reconv"]
SEVERITY_CLASSES = Literal["<",">"]


class Address(BaseModel):
    street: str = Field(
        description="""
            The street of the address. Can also be historical addresses, meaning they may no longer exist. Can also be the name of an institution or a place name. Written as the first, left-most, part of the address.
            Keep transcribing from left to right until you reach a street number or the edge of the page.
            Sometimes there is more text specifying the street, institution or place name in the row immediately below. If so, make sure to include it. Stop if you reach information relevant to hospital_stay.admission_date or fk_info.
        """
    )
    number: str = Field(
        description="Address number of the street. Written to the right of the street name. Often contains an integer and sometimes a letter. The letter always follows an integer. If it does not, it is not part of this variable."
    
    )
    apt: Optional[str] = Field(
        default=None,
        description="""
            Written right of the address number, describing the (danish-system) apartment floor. 
            Typically written with smaller font size. It's often a number, sometimes text, sometimes both. 
            Examples: [3, 1ste, St., kælderen, 2.o.G.]"""
    )

class Age(BaseModel):
    number: float = Field(
        description="""
            The age of the patient. After name. In some cases, there are multiple guesses of age, i.e. '2 (2.5?)'. 
            In the case of additional notation, only use the initial age mentioned (in this case, 2). 
            Can also include a division like 11/12, which then indicates a float-value. For 11/12 it would be 0.92.
        """
    )
    unit: str = Field(
        default="Aar",
        description="""
            The unit of the age, most often “aar” or “år” but can also be an indication of months, “mnd” or days “dage”. 
            Written next to the numerical value of the age.
        """
    )
    note: Optional[str] = Field(
        default=None,
        description="All additional text included after the age. Mostly used to capture uncertainty about the age. Examples: [Aar, (2.5?)]"
    )

class Patient(BaseModel):
    number: Optional[str] = Field(
        default=None,
        description="""
            The patient number in the center of the page, above the patient name. 
            Most often an integer, but can in rare cases include a letter as well.
        """
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
        description="Name of the doctor/physician conducting the journal. Usually as the last line in the bottom right of the page, but may also be right above the bottom diagnosis."
    )
    diagnosis: str = Field(
        description="A diagnosis found in the bottom right corner of the page, usually above the doctor's name, but can be the last row of the page as well. "
    )

class Sektion(BaseModel):
    number: int = Field(
        description="The dissection number, written at the top after “s.d.” or “sektionsdiagnose” or similar. Usually underlined and sometimes within parentheses. "
    )
    diagnoses: List[str] = Field(
        description="Consists of medical diagnoses or symptoms. Usually contains multiple items, each item defined by a new row. An item often includes a ditto dash, indicating that it is the same word as the one(s) immediately above it."
    )

class Ward(BaseModel):
    name: WARD_CLASSES = Field(
        description="""
            Name of the hospital ward. Strictly appears in the uppermost top-left corner.
            Often a combination of a letter and a number. The number can be given as either an Arabic or Roman numeral. 
            The ward can also be 'Observationsbygningen' or 'Officiantbygningen', often abbreviated as 'Observation',  'Obs.' or 'Officiant', Off.', respectively. The following rule exists, anything with Observation: 'Obs.' and anything regarding Officiant: 'Off.' 
            The ward can also be a tent ('Telt') which is often followed by a specifying letter. 
            Examples: {"Officiant":"Off.","Observation":"Obs.","FII":"F2","AI":"A1"}
        """
    )
    is_enestue: bool = Field(
        description="True if the patient has a private room. This is noted next to the ward variable. Examples: [”enestue”,”eneværelse”, “stue”, “st.”]"
    )

class HospitalStay(BaseModel):
    ward: Optional[Ward] = Field(
        default=None,
        description="The ward of the patient journal. Strictly written in the upper-most left corner, only the text in the very-upper corner, not anywhere else. Can be missing in rare cases."
    )
    admission_date: date = Field(
        ge=date(1879, 1, 1),
        le=date(1910, 12, 31),
        description="Written after 'Indl' or 'indskrevet' or similar indication of admission, and the upper-most date on the page. The year must be between 1879 and 1910."
    )
    release_date: date = Field(
        ge=date(1879, 1, 1),
        le=date(1910, 12, 31),
        description="Written after 'Udskr', 'Udskrevet' or similar indication of discharge, beneath admissionDate. May feature additional text indicating time of day. This is not transcribed here but in “note”. The year must be between 1879 and 1910."
    )
    stay_length: str = Field(
        description="Written beneath the release date. Commonly a number describing days, but sometimes also indication of hours (or very rarely, minutes) spent. Examples: [2 Dage (11 Timer), 5.5 Timer, 9 Dage, 3]"
    )
    note: Optional[str] = Field(
        default=None,
        description="Note for additional information after the length of stay, as, in rare cases, there may be additional text indicating time of day."
    )

class Top(BaseModel):
    conditions: List[str] = Field(
        description="""
        Usually found in the top right of the page and consists of medical diagnoses and symptoms. Usually it contains multiple items, each item defined by a new row. An item often includes a ditto dash, indicating that it is a continuation with the same word(s) as the one(s) immediately above it. 
        The list can be long and continue downwards adjacent to the patient's personal information. Make sure that no elements are cut off from the bottom of the list of diagnoses, but also that elements belonging to the patient's personal information are not included. 
    """
    )
    db: Optional[List[DB_CLASSES]] = Field(
        default=None,
        description="""
            If the patient is either positive or not positive for diphtheria bacteria, this has been noted either in the top-right corner or as part of the list of conditions.  
            Examples: {'+D.B.':'+DB', '÷D.B':'-DB', 'med D.B.':'+DB', 'ikke D.B.':'-DB', '+D.B. in reconv':'DB in reconv'}
            It may also include text after DB such as “in reconvalenscentia” or similar like “reconv”. Include in this variable all occurrences of this in a list. 
            The source generally uses '÷' in the absence of DB, this should always be mapped to '-'. 
        """
    )

class Severity(BaseModel):
    modifier: Optional[SEVERITY_CLASSES] = Field(
        default=None,
        description="A modifier that indicates if the given severity is worse than the word or better than the word. Written before the severity word. Sometimes written in the margin shadow."
    )
    word: str = Field(
        description="""
        A word indicating the severity of the patient. Often written on the left side of the page. 
        Examples: [spredt, middel, lidt, middelstærk, middelsvær, let, mild, svær, udbr., knapt middelstærk, belægninger] or a mix of these words. 
        Can also be 'moribund', 'haabløs' or similar where the severity modifier is often situated at the top of the page.
        """
    )

class Diagnoses(BaseModel):
    top: Top = Field(
        description="Diagnosis information generally found in the top middle and right part of the page."
    )
    bottom: Bottom = Field(
        description="Diagnosis and doctors name found in the bottom right corner of the page. "
    )
    sektion: Optional[Sektion] = Field(
        default=None,
        description="Only if is_dead is True. Usually found in the bottom half of the journal, below hospital_stay information. First row always contains an indication like: “s.d.” or “sektiondiagnosis”, “sdn”, or similar, followed by a number. "
    )
    severity: Optional[Severity] = Field(
        default=None,
        description="Found at the left side of the page, often in the middle, but can also be found elsewhere. Contains a word indicating the illness severity of the patient, and sometimes a modifier indicating if the severity is worse than or less than the word itself. The text can overlap with the margin in the far left side of the page, make sure to check there for parts of the string."
    )

class Serum(BaseModel):
    given: bool = Field(
        description="In the top left sometimes the test “serum” or similar is found."
    )
    doses: Optional[str] = Field(
        default=None,
        description=r"""
            Indicates the doses of serum given, as well as the days since symptom start. Only if SerumGiven is True. Usually an integer, or a list of integers separated by “+”. Can contain roman numerals or integers as well; often underneath the integers with in many cases a horizontal “{“ connecting subsequent doses on the same day, but can also be written underneath the integers without “{“. In nearly all cases these notes are written directly underneath the serum information (variable “type”), but can in rare cases extend on two or more lines. In rare cases, it can also be followed by a short string.  
            Should be transcribed where connected dosages are in square brackets and the roman numeral or integer is separated by a comma. Subsequent doses are added similarly.  
            Choose integers from 10, 15, 20, or 30. 
            For instance, 
            “[20]” is a single dose without further annotations 
            “[10, II]” is a single dose with roman numeral for 2 underneath. 
            “[20 + 10, V],[10, VI]” is three doses, on two days which are separated on the page by a horizontal “{“. 
            “[10, III],[20,IV]” is two doses on two days with the roman numeral for 3 and 4.  
            Examples = [[20],[10\,II],[20 + 10\, V]\,[10\, VI],[10\, III]\,[20\,IV]]
        """
    )
    type: Optional[str] = Field(
        default=None,
        description="The type of serum given. Usually contains the word ”Serum” followed by a specification such as “dan”, “dansk”, “danic”, “fransk”, “fra.”, “f”, “fort” or similar -- can be in different order."
    )


class FrontPage(BaseModel):
    is_dead: bool = Field(
        description="Determines whether the patient is dead. If true, a black cross is drawn on the page, typically at the top. Sometimes it looks like a plus, '+'. If the cross does not appear, it is False."
    )
    fk_info: str = Field(
        description="The page often contains an abbreviation in the left-center part. Typically this would say ”F.K.” or ”FK”, but alternative values include variations such as (but not limited to) ”u.k.”, ”udskr. u.k.” or ”e.O.” Specifications can appear in brackets, transcribe these as is. Do not include information that belongs in diagnoses.severity such as ”middel” or ”svær”."
    )
    patient: Patient = Field(
        description="Patient related information like, name, age, address, position in the household and patient number. Generally located around the center of the page. The information is written in late 19th and early 20th century Danish."
    )
    hospital_stay: HospitalStay = Field(
        description="Contains information about the patient's stay at the hospital, like the ward, the admission and discharge dates, and the length of stay. Most information is found underneath the patient information, but ward can be found in the top left corner. Also contains severity information, usually found on the left side of the page and indicating the general severity of the patient. The information is written in late 19th and early 20th century Danish."
    )
    diagnoses: Diagnoses = Field(
        description="Contains all information related to the diagnoses. Contains mainly medical information and terminology. It is divided into several parts: top diagnoses written in the top right comer of the page; bottom diagnoses written in the bottom right comer—which also include the doctor's name; and section diagnoses which are generally found in the (bottom) left part of the page.  The information is written in late 19th and early 20th century Danish and often contains greek / latin medical descriptions of symptoms and diagnoses. "
    )
    serum: Serum = Field(
        description="Contains information on whether the patient was treated with anti-diphtheria serum, and if so, the type of serum and dosis information. Not all cases have this information. If it is found, it is in the top left corner of the journal, underneath or beside the ward information. Often written diagonally."
    )
    crossed_out: Optional[str] = Field(
        default=None,
        description="Verbose description of any sections of the page that have crossed out (not underlined) words or numbers, that have not been included in any other variables."
    )


# text pages schema
class PageLine(BaseModel):
    text: str = Field(description="This includes all text described that isn't written seperated from the line.")
    metadata: Optional[str] = Field(None, description="Contains the metadata of the line, can describe dates, temperatures, etc. Most commonly written in the left-side, before normal text is written.")
    page_line_number: Optional[int] = Field(
        default=None,
        description="1-based line number derived from this item's position in page_lines.",
    )


class TextPage(BaseModel):
    page_lines: List[PageLine] = Field(description="This is meant for each line on the page, seperated by linebreaks.")

    @model_validator(mode="after")
    def assign_line_numbers(self) -> "TextPage":
        for index, line in enumerate(self.page_lines, start=1):
            line.page_line_number = index
        return self
