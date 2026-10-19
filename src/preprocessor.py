"""
Feature definitions, preprocessing pipeline construction, and metadata schema.
"""

from typing import List, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Explicit, ordered list of categorical features (17 features)
CATEGORICAL_FEATURES: List[str] = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

# Explicit, ordered list of numerical / ordinal features (15 features)
NUMERICAL_FEATURES: List[str] = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
]

# Complete feature order expected by the model (32 features)
FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Target classes in order
TARGET_CLASSES: List[str] = [
    "Low Performance",
    "Medium Performance",
    "High Performance",
]

# Human-readable metadata and options for UI forms and data validation
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "school": {
        "label": "School",
        "type": "categorical",
        "options": ["GP", "MS"],
        "labels": {"GP": "Gabriel Pereira (GP)", "MS": "Mousinho da Silveira (MS)"},
        "default": "GP",
        "description": "Student's secondary school",
    },
    "sex": {
        "label": "Sex",
        "type": "categorical",
        "options": ["F", "M"],
        "labels": {"F": "Female", "M": "Male"},
        "default": "F",
        "description": "Student's sex",
    },
    "age": {
        "label": "Age",
        "type": "numerical",
        "min": 15,
        "max": 22,
        "default": 17,
        "description": "Student's age in years (15 - 22)",
    },
    "address": {
        "label": "Address Type",
        "type": "categorical",
        "options": ["U", "R"],
        "labels": {"U": "Urban", "R": "Rural"},
        "default": "U",
        "description": "Home address zone",
    },
    "famsize": {
        "label": "Family Size",
        "type": "categorical",
        "options": ["GT3", "LE3"],
        "labels": {"GT3": "Greater than 3 members", "LE3": "3 members or fewer"},
        "default": "GT3",
        "description": "Family household size",
    },
    "Pstatus": {
        "label": "Parent Cohabitation Status",
        "type": "categorical",
        "options": ["T", "A"],
        "labels": {"T": "Living Together", "A": "Apart / Divorced"},
        "default": "T",
        "description": "Parents' relationship living arrangement",
    },
    "Medu": {
        "label": "Mother's Education Level",
        "type": "ordinal",
        "options": [0, 1, 2, 3, 4],
        "labels": {
            0: "0 - None",
            1: "1 - Primary education (4th grade)",
            2: "2 - 5th to 9th grade",
            3: "3 - Secondary education (10-12th grade)",
            4: "4 - Higher / University education",
        },
        "default": 2,
        "description": "Highest formal education attained by mother",
    },
    "Fedu": {
        "label": "Father's Education Level",
        "type": "ordinal",
        "options": [0, 1, 2, 3, 4],
        "labels": {
            0: "0 - None",
            1: "1 - Primary education (4th grade)",
            2: "2 - 5th to 9th grade",
            3: "3 - Secondary education (10-12th grade)",
            4: "4 - Higher / University education",
        },
        "default": 2,
        "description": "Highest formal education attained by father",
    },
    "Mjob": {
        "label": "Mother's Occupation",
        "type": "categorical",
        "options": ["at_home", "health", "services", "teacher", "other"],
        "labels": {
            "at_home": "At Home / Homemaker",
            "health": "Healthcare professional",
            "services": "Civil / Administrative services",
            "teacher": "Teacher / Educator",
            "other": "Other employment",
        },
        "default": "other",
        "description": "Mother's employment field",
    },
    "Fjob": {
        "label": "Father's Occupation",
        "type": "categorical",
        "options": ["at_home", "health", "services", "teacher", "other"],
        "labels": {
            "at_home": "At Home / Homemaker",
            "health": "Healthcare professional",
            "services": "Civil / Administrative services",
            "teacher": "Teacher / Educator",
            "other": "Other employment",
        },
        "default": "other",
        "description": "Father's employment field",
    },
    "reason": {
        "label": "Reason for Choosing School",
        "type": "categorical",
        "options": ["course", "home", "reputation", "other"],
        "labels": {
            "course": "Course curriculum preference",
            "home": "Close to home",
            "reputation": "School academic reputation",
            "other": "Other reason",
        },
        "default": "course",
        "description": "Primary factor for choosing the school",
    },
    "guardian": {
        "label": "Legal Guardian",
        "type": "categorical",
        "options": ["mother", "father", "other"],
        "labels": {"mother": "Mother", "father": "Father", "other": "Other legal guardian"},
        "default": "mother",
        "description": "Student's legal guardian",
    },
    "traveltime": {
        "label": "Travel Time to School",
        "type": "ordinal",
        "options": [1, 2, 3, 4],
        "labels": {
            1: "< 15 minutes",
            2: "15 to 30 minutes",
            3: "30 minutes to 1 hour",
            4: "> 1 hour",
        },
        "default": 1,
        "description": "Daily one-way commute duration to school",
    },
    "studytime": {
        "label": "Weekly Study Time",
        "type": "ordinal",
        "options": [1, 2, 3, 4],
        "labels": {
            1: "< 2 hours / week",
            2: "2 to 5 hours / week",
            3: "5 to 10 hours / week",
            4: "> 10 hours / week",
        },
        "default": 2,
        "description": "Average hours spent studying per week",
    },
    "failures": {
        "label": "Past Course Failures",
        "type": "numerical",
        "min": 0,
        "max": 4,
        "default": 0,
        "description": "Number of past class failures (0 to 4)",
    },
    "schoolsup": {
        "label": "Extra School Educational Support",
        "type": "categorical",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Yes"},
        "default": "no",
        "description": "Extra academic tutoring provided by school",
    },
    "famsup": {
        "label": "Family Educational Support",
        "type": "categorical",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Yes"},
        "default": "yes",
        "description": "Family provides educational assistance or tutoring",
    },
    "paid": {
        "label": "Extra Paid Classes",
        "type": "categorical",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Yes"},
        "default": "no",
        "description": "Enrolled in paid outside tutoring classes for math",
    },
    "activities": {
        "label": "Extracurricular Activities",
        "type": "categorical",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Yes"},
        "default": "no",
        "description": "Participates in sports, arts, or school clubs",
    },
    "nursery": {
        "label": "Attended Nursery / Preschool",
        "type": "categorical",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Yes"},
        "default": "yes",
        "description": "Attended early childhood preschool",
    },
    "higher": {
        "label": "Desires Higher Education",
        "type": "categorical",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Yes"},
        "default": "yes",
        "description": "Aspires to pursue university / college education",
    },
    "internet": {
        "label": "Home Internet Access",
        "type": "categorical",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Yes"},
        "default": "yes",
        "description": "Availability of high-speed internet at home",
    },
    "romantic": {
        "label": "In a Romantic Relationship",
        "type": "categorical",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Yes"},
        "default": "no",
        "description": "Currently in a romantic relationship",
    },
    "famrel": {
        "label": "Family Relationship Quality",
        "type": "ordinal",
        "options": [1, 2, 3, 4, 5],
        "labels": {
            1: "1 - Very bad",
            2: "2 - Poor",
            3: "3 - Average",
            4: "4 - Good",
            5: "5 - Excellent",
        },
        "default": 4,
        "description": "Subjective quality of family relationships",
    },
    "freetime": {
        "label": "Free Time After School",
        "type": "ordinal",
        "options": [1, 2, 3, 4, 5],
        "labels": {
            1: "1 - Very low",
            2: "2 - Low",
            3: "3 - Moderate",
            4: "4 - High",
            5: "5 - Very high",
        },
        "default": 3,
        "description": "Available free time after school hours",
    },
    "goout": {
        "label": "Socializing / Going Out with Friends",
        "type": "ordinal",
        "options": [1, 2, 3, 4, 5],
        "labels": {
            1: "1 - Very rarely",
            2: "2 - Rarely",
            3: "3 - Sometimes",
            4: "4 - Frequently",
            5: "5 - Very frequently",
        },
        "default": 3,
        "description": "Frequency of going out with peers",
    },
    "Dalc": {
        "label": "Workday Alcohol Consumption",
        "type": "ordinal",
        "options": [1, 2, 3, 4, 5],
        "labels": {
            1: "1 - Very low / None",
            2: "2 - Low",
            3: "3 - Moderate",
            4: "4 - High",
            5: "5 - Very high",
        },
        "default": 1,
        "description": "Workday alcohol consumption level",
    },
    "Walc": {
        "label": "Weekend Alcohol Consumption",
        "type": "ordinal",
        "options": [1, 2, 3, 4, 5],
        "labels": {
            1: "1 - Very low / None",
            2: "2 - Low",
            3: "3 - Moderate",
            4: "4 - High",
            5: "5 - Very high",
        },
        "default": 1,
        "description": "Weekend alcohol consumption level",
    },
    "health": {
        "label": "Current Health Status",
        "type": "ordinal",
        "options": [1, 2, 3, 4, 5],
        "labels": {
            1: "1 - Very poor",
            2: "2 - Fair",
            3: "3 - Good",
            4: "4 - Very good",
            5: "5 - Excellent",
        },
        "default": 4,
        "description": "Current overall health rating",
    },
    "absences": {
        "label": "School Absences (Days)",
        "type": "numerical",
        "min": 0,
        "max": 93,
        "default": 4,
        "description": "Total number of school days missed",
    },
    "G1": {
        "label": "Period 1 Grade (G1)",
        "type": "numerical",
        "min": 0,
        "max": 20,
        "default": 11,
        "description": "First period academic grade on Portuguese 0-20 scale",
    },
    "G2": {
        "label": "Period 2 Grade (G2)",
        "type": "numerical",
        "min": 0,
        "max": 20,
        "default": 11,
        "description": "Second period academic grade on Portuguese 0-20 scale",
    },
}


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-learn ColumnTransformer:
    - Standardizes numerical/ordinal features with StandardScaler.
    - One-hot encodes categorical features with handle_unknown='ignore'.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor
