import os
import cv2
import sys
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
from glob import glob
from shutil import copyfile
from matplotlib import patches
from sklearn import preprocessing, model_selection


def x_center(x: pd.Series) -> int:
    return int(x.x_scaled + (x.w_scaled / 2))


def y_center(y: pd.Series) -> int:
    return int(y.y_scaled + (y.h_scaled / 2))


def w_normalise(w: pd.Series, column: str) -> float:
    return w[column] / w['page_width_scaled']


def h_normalise(h: pd.Series, column: str) -> float:
    return h[column] / h['page_height_scaled']


def extract_info_from_xml(path_to_xml_folder: str) -> pd.DataFrame:
    if not os.path.exists(path_to_xml_folder):
        raise Exception("Path of the folder is invalid")
    columns = ['prev_filename', 'filename', 'page_height', 'page_width', 'AuthorID', 'Overlapped', 'category', 'id',
               'x', 'y', 'width', 'height']
    df = []
    cnt = 0
    for file in tqdm(sorted(glob(f"{path_to_xml_folder}/*.xml")), desc="Extracting info from dataset 'Tobacco800'"):
        my_root = ET.parse(file).getroot()
        prev_filename = my_root[0].attrib['src']
        filename = f"{str(cnt)}.tif"
        page_height, page_width = my_root[0][0].attrib['height'], my_root[0][0].attrib['width']
        # An image might have multiple items (zones) (logos and signs), so iterate through each zones
        for zone in my_root[0][0]:
            category = zone.attrib['gedi_type']  # type of zone (DLLogo/ DLSignature)
            id = zone.attrib['id']
            x, y = zone.attrib['col'], zone.attrib['row']  # x, y coordinate
            w, h = zone.attrib['width'], zone.attrib['height']  # width and height of bbox
            if category == 'DLSignature':  # Signature have Authors, representing whose signature it is
                author_id, overlapped = (zone.attrib['AuthorID'], zone.attrib['Overlapped'])
            else:  # Logos don't have authors.
                author_id, overlapped = ('NA', 'NA')
            row = [prev_filename, filename, page_height, page_width, author_id, overlapped, category, id, x, y, w, h]
            df.append(row)
        cnt += 1
    return pd.DataFrame(df, columns=columns)


def scale_dataset_images(dataframe: pd.DataFrame, path_to_tif_files: str, output_path_to_images: str) -> pd.DataFrame:
    if not os.path.exists(path_to_tif_files):
        raise Exception("Path of the folder 'path_to_tif_files' is invalid")
    if not os.path.exists(output_path_to_images):
        raise Exception("Path of the folder 'output_path_to_images' is invalid")
    filename = dataframe.prev_filename
    X, Y = map(int, dataframe.x), map(int, dataframe.y)
    W, H = map(int, dataframe.width), map(int, dataframe.height)
    dataframe['new_filename'] = ""
    dataframe['x_scaled'] = np.nan
    dataframe['y_scaled'] = np.nan
    dataframe['w_scaled'] = np.nan
    dataframe['h_scaled'] = np.nan
    dataframe['page_height_scaled'] = np.nan
    dataframe['page_width_scaled'] = np.nan
    for file, x, y, w, h in tqdm(zip(filename, X, Y, W, H), desc="Scaling dataset 'Tobacco800'"):
        img = cv2.imread(os.path.join(path_to_tif_files, file), 1)
        page_height, page_width = img.shape[:2]
        max_height = 640
        max_width = 480
        if max_height < page_height or max_width < page_width:
            scaling_factor = max_height / float(page_height)
            if max_width / float(page_width) < scaling_factor:
                scaling_factor = max_width / float(page_width)
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        jpg_filename = f"{file[:-4]}.jpg"
        cv2.imwrite(os.path.join(output_path_to_images, jpg_filename), img)  # write the scales image
        try:
            index = dataframe[dataframe['prev_filename'] == file].index.item()
        except:
            index = 0
            indexes = dataframe[dataframe['prev_filename'] == file].index.to_list()
            for i in indexes:
                if dataframe.at[i, 'new_filename'] == "":
                    index = i
                    break
        dataframe.at[index, 'new_filename'] = jpg_filename
        dataframe.at[index, 'x_scaled'] = int(x * scaling_factor)
        dataframe.at[index, 'y_scaled'] = int(y * scaling_factor)
        dataframe.at[index, 'w_scaled'] = int(w * scaling_factor)
        dataframe.at[index, 'h_scaled'] = int(h * scaling_factor)
        dataframe.at[index, 'page_height_scaled'] = page_height * scaling_factor
        dataframe.at[index, 'page_width_scaled'] = page_width * scaling_factor
    return dataframe


def segregate_data(dataframe: pd.DataFrame,
                   path_to_images: str, path_to_xml_files: str,
                   path_to_output_images: str, path_to_output_labels: str) -> None:
    if not os.path.exists(path_to_images):
        raise Exception("Path of the folder 'path_to_images' is invalid")
    if not os.path.exists(path_to_xml_files):
        raise Exception("Path of the folder 'path_to_xml_files' is invalid")

    if not os.path.exists(path_to_output_images):
        raise Exception("Path of the folder 'path_to_output_images' is invalid")
    if not os.path.exists(path_to_output_labels):
        raise Exception("Path of the folder 'path_to_output_labels' is invalid")

    for filename in tqdm(set([filename for filename in dataframe.filename]), desc="Converting data to YOLO model..."):
        yolo_list = []
        for _, row in dataframe[dataframe.filename == filename].iterrows():
            yolo_list.append([row.labels, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])
        yolo_list = np.array(yolo_list)
        txt_filename = os.path.join(path_to_output_labels, f"{str(row.new_filename.split('.')[0])}.txt")
        np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
        shutil.copyfile(os.path.join(path_to_images, row.new_filename),
                        os.path.join(path_to_output_images, row.new_filename))


OUTPUT_PATH_TO_IMAGES = '../dataset/scaled'
OUTPUT_PATH_TO_DATA = '../dataset/data.csv'
BASE_TO_TIF_FILES = r'../tobacco_data_zhugy/Tobacco800_SinglePage/Tobacco800_SinglePage/SinglePageTIF'
PAHT_TO_XML_FOLDER = "../tobacco_data_zhugy/Tobacc800_Groundtruth_v2.0/Tobacc800_Groundtruth_v2.0/XMLGroundtruth_v2.0"

for path in ['../dataset/', '../dataset/scaled/',
             '../dataset/images/', '../dataset/images/train/', '../dataset/images/valid/',
             '../dataset/labels/', '../dataset/labels/train/', '../dataset/labels/valid/']:
    try:
        os.mkdir(path)
    except:
        pass

data = extract_info_from_xml(path_to_xml_folder=PAHT_TO_XML_FOLDER)
data = scale_dataset_images(dataframe=data,
                            path_to_tif_files=BASE_TO_TIF_FILES,
                            output_path_to_images=OUTPUT_PATH_TO_IMAGES)

label_encoder = preprocessing.LabelEncoder().fit(data['category'])
data['labels'] = label_encoder.transform(data['category'])
data['x_center'] = data.apply(x_center, axis=1)
data['y_center'] = data.apply(y_center, axis=1)
data['x_center_norm'] = data.apply(w_normalise, column='x_center', axis=1)
data['width_norm'] = data.apply(w_normalise, column='w_scaled', axis=1)
data['y_center_norm'] = data.apply(h_normalise, column='y_center', axis=1)
data['height_norm'] = data.apply(h_normalise, column='h_scaled', axis=1)
data.to_csv(OUTPUT_PATH_TO_DATA)
data_train, data_valid = model_selection.train_test_split(data, test_size=0.1, random_state=13, shuffle=True)

segregate_data(dataframe=data_train,
               path_to_images=OUTPUT_PATH_TO_IMAGES,
               path_to_xml_files=PAHT_TO_XML_FOLDER,
               path_to_output_images='../dataset/images/train',
               path_to_output_labels='../dataset/labels/train')
segregate_data(dataframe=data_valid,
               path_to_images=OUTPUT_PATH_TO_IMAGES,
               path_to_xml_files=PAHT_TO_XML_FOLDER,
               path_to_output_images='../dataset/images/valid',
               path_to_output_labels='../dataset/labels/valid')

print("No. of Training images", len(os.listdir('../dataset/images/train')))
print("No. of Training labels", len(os.listdir('../dataset/labels/train')))
print("No. of valid images", len(os.listdir('../dataset/images/valid')))
print("No. of valid labels", len(os.listdir('../dataset/labels/valid')))

