# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

"""


########################################################################################################################
# Handling the imports
########################################################################################################################
import json
import sys
import os
import torch

# Loading the config file to get the parameters
with open('config.json') as f:
    config = json.load(f)

_RAUG_PATH = config['raug_path']
_METABLOCK_PATH = config['metablock_path']
_CHECKPOINT_PATH = config['checkpoint_path']

sys.path.insert(0, _RAUG_PATH)
sys.path.insert(0, _METABLOCK_PATH)
sys.path.insert(0, os.path.join(_METABLOCK_PATH, "my_models"))
sys.path.insert(0, os.path.join(_METABLOCK_PATH, "benchmarks", "pad"))

from raug.checkpoints import load_model
from raug.eval import test_single_input
from my_model import set_model
from aug_pad import ImgEvalTransform
import torch.nn.functional as nnF
import onnxruntime as ort
import numpy as np
########################################################################################################################

########################################################################################################################
# Constants / Default configs
# Don't change this if you don't know what you are doing
########################################################################################################################
_LABELS_NAME = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
_MODEL_NAME = 'resnet-50'
_NEURONS_REDUCER_BLOCK = 0
_COMB_METHOD = "metablock"
_COMB_CONFIG = [64, 45]
########################################################################################################################

########################################################################################################################
# Defining the functions
########################################################################################################################
def _set_meta_data(body_region, fitzpatrick, age, gender, smoke, drink, itch, growth, bled, pain, changed,
                   elevation, cancer_history, skin_cancer_history):
    """
    This function sets the metadata to be used in the model inference. It get the inputs from the interface, convert
    them to the model's format and return a list with the metadata. It is a "private" function, so it should not be
    called outside this file.

    Note: this version of the model is not using smoke and drink. However, the interface is collecting it. So,
    I left it here for future use
    """

    meta = {
        "age": int(age),
        "gender_FEMALE": 1 if gender == "Feminino" else 0,
        "gender_MALE": 1 if gender == "Masculino" else 0,
        "skin_cancer_history_True": 1 if skin_cancer_history == "Sim" else 0,
        "skin_cancer_history_False": 1 if skin_cancer_history == "Não" else 0,
        "cancer_history_True": 1 if cancer_history == "Sim" else 0,
        "cancer_history_False": 1 if cancer_history == "Não" else 0,
        "fitspatrick_3.0": 1 if fitzpatrick == "Tipo III - Pele branca, cabelos e olhos de outras cores que não são claros" else 0,
        "fitspatrick_1.0": 1 if fitzpatrick == "Tipo I - Pele muito branca, sardas, cabelo loiro ou ruivo, olhos claros" else 0,
        "fitspatrick_2.0": 1 if fitzpatrick == "Tipo II - Pele branca, cabelo loiros, olhos claros" else 0,
        "fitspatrick_4.0": 1 if fitzpatrick == "Tipo IV - Pele moderadamente pigmentada ou pele morena clara" else 0,
        "fitspatrick_5.0": 1 if fitzpatrick == "Tipo V - Pele escura" else 0,
        "fitspatrick_6.0": 1 if fitzpatrick == "Tipo VI - Pele muito escura" else 0,
        "region_ARM": 1 if body_region == "Braço" else 0,
        "region_NECK": 1 if body_region == "Pescoço" else 0,
        "region_FACE": 1 if body_region == "Face/Rosto" else 0,
        "region_HAND": 1 if body_region == "Mão" else 0,
        "region_FOREARM": 1 if body_region == "Antebraço" else 0,
        "region_CHEST": 1 if body_region == "Peito" else 0,
        "region_NOSE": 1 if body_region == "Nariz" else 0,
        "region_THIGH": 1 if body_region == "Coxa" else 0,
        "region_SCALP": 1 if body_region == "Couro cabeludo" else 0,
        "region_EAR": 1 if body_region == "Orelha" else 0,
        "region_BACK": 1 if body_region == "Costas" else 0,
        "region_FOOT": 1 if body_region == "Pé" else 0,
        "region_ABDOMEN": 1 if body_region == "Abdome" else 0,
        "region_LIP": 1 if body_region == "Lábios" else 0,
        "itch_False": 1 if itch == "Não" else 0,
        "itch_True": 1 if itch == "Sim" else 0,
        "itch_UNK": 1 if itch == "Não Sabe/Ignorado" else 0,
        "grew_False": 1 if growth == "Não" else 0,
        "grew_True": 1 if growth == "Sim" else 0,
        "grew_UNK": 1 if growth == "Não Sabe/Ignorado" else 0,
        "hurt_False": 1 if pain == "Não" else 0,
        "hurt_True": 1 if pain == "Sim" else 0,
        "hurt_UNK": 1 if pain == "Não Sabe/Ignorado" else 0,
        "changed_False": 1 if changed == "Não" else 0,
        "changed_True": 1 if changed == "Sim" else 0,
        "changed_UNK": 1 if changed == "Não Sabe/Ignorado" else 0,
        "bleed_False": 1 if bled == "Não" else 0,
        "bleed_True": 1 if bled == "Sim" else 0,
        "bleed_UNK": 1 if bled == "Não Sabe/Ignorado" else 0,
        "elevation_False": 1 if elevation == "Não" else 0,
        "elevation_True": 1 if elevation == "Sim" else 0,
        "elevation_UNK": 1 if elevation == "Não Sabe/Ignorado" else 0
    }

    return list(meta.values())


def set_and_load_model():
    """
    This function sets the model we're going to use for inference and load the checkpoints for it.
    If everything goes well, it returns the model.
    """

    # Setting the model we're going to use for inference
    model = set_model(_MODEL_NAME, len(_LABELS_NAME), neurons_reducer_block=_NEURONS_REDUCER_BLOCK,
                      comb_method=_COMB_METHOD, comb_config=_COMB_CONFIG, pretrained=False)

    # Loading the checkpoints for the model
    model = load_model(_CHECKPOINT_PATH, model)
    model.eval()

    return model


def make_inference(model, img, body_region, fitzpatrick, age, gender, smoke, drink, itch, growth, bled, pain, changed,
                   elevation, cancer_history, skin_cancer_history, verbose=False):
    """
        This function makes the inference for the given image and metadata.
        If verbose is True, it prints the results in the terminal.
        It returns the predicted label and the predicted probability for it.
    """
    # Setting the metadata according to the inputs
    metadata = _set_meta_data(body_region, fitzpatrick, age, gender, smoke, drink, itch, growth, bled, pain, changed,
                   elevation, cancer_history, skin_cancer_history)

    # Setting the image transform to the same used in model's training
    _trans = ImgEvalTransform()

    # Making the inference
    pred = test_single_input(model, _trans, img, metadata)[0]
    pred_label = _LABELS_NAME[pred.argmax()]
    pred_prob = pred.max()

    # Showing the results in the terminal if verbose is True
    if verbose:
        print("-" * 50)
        print("- Predição:")
        print("-" * 50)
        print(f"- Lesão: {_LABELS_NAME[pred.argmax()]}")
        print("- Probabilidades:")
        for l, p in zip(_LABELS_NAME, pred):
            print(f"-- {l}: {100 * p:.2f}%")
        print("-" * 50, "\n")


    # Daqui para baixo é apenas uma demostração que funciona o ONNX. É necessários realizar mais testes
    # para finalizar a implementação e alterar a documentação.
    # MODEL = ort.InferenceSession("/home/apacheco/code/MetaBlock/benchmarks/pad/resnet-50-45-meta.onnx")

    # img = _trans(img).unsqueeze(0).numpy() # adding the batch size dimension (which must be 1)    
    # meta_data = np.array(metadata)
    # meta_data = np.expand_dims(meta_data, axis=0) # idem

    # onnx_inputs = {'img_input': img, 'meta_data_input': meta_data}
    # y = MODEL.run(None, onnx_inputs)[0]    
    # _pred = nnF.softmax(torch.from_numpy(y), dim=1).numpy()[0]    
    
    # if verbose:
    #     print("-" * 50)
    #     print("- Predição ONNX:")
    #     print("-" * 50)
    #     print(f"- Lesão: {_LABELS_NAME[_pred.argmax()]}")
    #     print("- Probabilidades ONXX:")
    #     for l, p in zip(_LABELS_NAME, _pred):
    #         print(f"-- {l}: {100 * p:.2f}%")
    #     print("-" * 50, "\n")

    return pred, pred_label, pred_prob

def make_batch_inference(model, images, metadata_list, batch_size=32, device=None):
    """
    Esta função realiza inferências em batch para um conjunto de imagens e metadados.

    :param model: Modelo carregado que será utilizado para inferência.
    :param images: Lista de imagens PIL a serem avaliadas.
    :param metadata_list: Lista de dicionários de metadados para cada imagem.
    :param batch_size: Tamanho do lote para a inferência.
    :param device: Dispositivo a ser usado ('cuda' ou 'cpu').
    :return: Lista de previsões para cada imagem.
    """

    # Definir o dispositivo
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configurar o modelo para o modo de avaliação e movê-lo para o dispositivo
    model.eval()
    model.to(device)

    # Configurar a transformação de imagem usada durante o treinamento
    _trans = ImgEvalTransform()

    predictions = []

    # Fazer inferência em lotes
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_metadata = metadata_list[i:i+batch_size]

            # Transformar imagens e preparar metadados para o modelo
            imgs_tensor = torch.stack([_trans(img) for img in batch_images]).to(device)
            metadata_tensor = torch.tensor([list(meta.values()) for meta in batch_metadata], dtype=torch.float).to(device)

            # Fazer a inferência com o modelo
            batch_preds = model(imgs_tensor, metadata_tensor)

            # Aplicar softmax para obter as probabilidades e movê-las para a CPU
            batch_preds = torch.nn.functional.softmax(batch_preds, dim=1).cpu().numpy()

            # Adicionar as previsões ao resultado final
            predictions.extend(batch_preds)

    return predictions

