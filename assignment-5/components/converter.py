sex_types = {
    "Male": 1,
    "Female": 0
}

thal_types = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2
}

cp_types = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

slope_types = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

resting_ecg_types = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

boolean_types = {
    "Yes": 1,
    "No": 0
}

def convert_input(input_data: dict) -> dict:
    convert_cols = {
        "sex": sex_types, 
        "cp": cp_types, 
        "thal": thal_types, 
        "slope": slope_types,
        "restecg": resting_ecg_types, 
        "fbs": boolean_types, 
        "exang": boolean_types
    }
    converted_input = {}
    
    for key, value in input_data.items():
        if key in convert_cols:
            converted_value = convert_cols.get(key).get(value)
        else:
            converted_value = value
        converted_input.setdefault(key, converted_value)
    
    return converted_input
