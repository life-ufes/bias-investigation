def map_metadata(random_row):
    """ Convert metadata attributes to their respective mapped values """
    boolean_map = {True: "Sim", False: "Não", "True": "Sim", "False": "Não"}
    gender_map = {"MALE": "Masculino", "FEMALE": "Feminino"}
    fitzpatrick_map = {
        1: "Tipo I - Pele muito branca, sardas, cabelo loiro ou ruivo, olhos claros",
        2: "Tipo II - Pele branca, cabelo loiros, olhos claros",
        3: "Tipo III - Pele branca, cabelos e olhos de outras cores que não são claros",
        4: "Tipo IV - Pele moderadamente pigmentada ou pele morena clara",
        5: "Tipo V - Pele escura",
        6: "Tipo VI - Pele muito escura"
    }
    region_map = {
        "SCALP": "Couro cabeludo",
        "NOSE": "Nariz",
        "EAR": "Orelha",
        "LIPS": "Lábios",
        "FACE": "Face/Rosto",
        "NECK": "Pescoço",
        "BACK": "Costas/Dorso",
        "CHEST": "Peito",
        "ABDOMEN": "Abdome",
        "ARM": "Braço",
        "FOREARM": "Antebraço",
        "HAND": "Mão",
        "THIGH": "Coxa",
        "LEG": "Perna/Canela",
        "FOOT": "Pé",
        "UNKNOWN": "Não Sabe/Ignorado"
    }

    return {
        "region": region_map.get(random_row['region'].upper(), random_row['region']),
        "fitzpatrick": fitzpatrick_map.get(random_row['fitspatrick'], random_row['fitspatrick']),
        "gender": gender_map.get(random_row['gender'], random_row['gender']),
        "smoke": boolean_map.get(random_row['smoke'], random_row['smoke']),
        "drink": boolean_map.get(random_row['drink'], random_row['drink']),
        "itch": boolean_map.get(random_row['itch'], random_row['itch']),
        "grew": boolean_map.get(random_row['grew'], random_row['grew']),
        "bleed": boolean_map.get(random_row['bleed'], random_row['bleed']),
        "hurt": boolean_map.get(random_row['hurt'], random_row['hurt']),
        "changed": boolean_map.get(random_row['changed'], random_row['changed']),
        "elevation": boolean_map.get(random_row['elevation'], random_row['elevation']),
        "cancer_history": boolean_map.get(random_row['cancer_history'], random_row['cancer_history']),
        "skin_cancer_history": boolean_map.get(random_row['skin_cancer_history'], random_row['skin_cancer_history']),
        "age": random_row['age']
    }
