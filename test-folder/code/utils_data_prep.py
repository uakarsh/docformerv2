import os
import requests
from pathlib import Path
import io
from PyPDF2 import PdfReader
import json
from typing import List
from PIL import Image

def download_and_store_pdf(file_path) -> int:
    '''
    args: file_path: file path corresponding to the OCR stored in json
    returns: None, and stores the corresponding pdf in the same folder structure as that of OCR
    '''
    
    url = "https://download.industrydocuments.ucsf.edu"
    sample_file = file_path
    actual_path = sample_file.replace('ocrs', 'pdfs').split("/")[:-1]
    file_name = sample_file.split("/")[-1].split(".")[0] + ".pdf"
    dest_path = "/".join(actual_path) if file_path[0] == '/' else os.path.join(*actual_path)
    os.makedirs(dest_path, exist_ok=True)
    dest_path = os.path.join(dest_path, file_name)

    if os.path.exists(dest_path):
        return 1

    idx = sample_file.split("/").index("ocrs")
    for i in actual_path[idx+1:]:
        url = url + "/" + i
    url = url + "/" + actual_path[-1]
    url = url + ".pdf"

    try:
        response = requests.get(url)
        filename = Path(dest_path)
        filename.write_bytes(response.content)
        return 1
    except:
        return 0
    

# Image property
resize_scale = (500, 500)

def normalize_box(box: List[int], width: int, height: int, size: tuple = resize_scale):
    """
    Takes a bounding box and normalizes it to a thousand pixels. If you notice it is
    just like calculating percentage except takes 1000 instead of 100.
    """
    return [
        int(size[0] * (box[0] / width)),
        int(size[1] * (box[1] / height)),
        int(size[0] * (box[2] / width)),
        int(size[1] * (box[3] / height)),
    ]


# Function to get the images from the PDFs as well as the OCRs for the corresponding images
def get_image_ocrs_from_path(pdf_file_path: str, ocr_file_path: str, resize_scale=resize_scale,
                             save_folder_img: str = "../data/images", save_folder_ocr: str = "../data/ocrs"):

    # Making folder to save the images
    if not os.path.exists(save_folder_img):
        os.mkdir(save_folder_img)

    # Making folder to save the OCRs
    if not os.path.exists(save_folder_ocr):
        os.mkdir(save_folder_ocr)

    try:

        # Getting the image list, since the pdfs can contain many image
        reader = PdfReader(pdf_file_path)
        img_list = {}
        pg_count = 1
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            for image_file_object in page.images:

                stream = io.BytesIO(image_file_object.data)
                img = Image.open(stream).convert("RGB").resize(resize_scale)
                path_name = os.path.join(
                    save_folder_img, f"{pdf_file_path.split('/')[-1].split('.')[0]}_{pg_count}.png")
                pg_count += 1
                img.save(path_name)
                img_list[pg_count - 1] = path_name

        json_entry = json.load(open(ocr_file_path))[1]
        json_entry = [x for x in json_entry["Blocks"] if "Text" in x]

        pages = [x["Page"] for x in json_entry]
        ocrs = {pg: [] for pg in set(pages)}

        for entry in json_entry:
            bbox = entry["Geometry"]["BoundingBox"]
            x, y, w, h = bbox['Left'], bbox['Top'], bbox["Width"], bbox["Height"]
            bbox = [x, y, x + w, y + h]
            # bbox = normalize_box(bbox, width=1, height=1, size=resize_scale)
            ocrs[entry["Page"]].append({"word": entry["Text"], "bbox": bbox})

        ocr_path = {}
        for pg in set(pages):
            path_name = os.path.join(
                save_folder_ocr, f"{pdf_file_path.split('/')[-1].split('.')[0]}_{pg}.json")
            with open(path_name, "w") as f:
                json.dump(ocrs[pg], f)
            ocr_path[pg] = path_name

        return img_list, ocr_path

    except:
        return {}, {}


# Function to get the OCRs for the corresponding images
def get_ocrs_from_path(ocr_file_path: str, save_folder_ocr: str = "./ocrs"):

    # Making folder to save the OCRs
    if not os.path.exists(save_folder_ocr):
        os.mkdir(save_folder_ocr)

    try:

        json_entry = json.load(open(ocr_file_path))[1]
        json_entry = [x for x in json_entry["Blocks"] if "Text" in x]

        pages = [x["Page"] for x in json_entry]
        ocrs = {pg: [] for pg in set(pages)}

        for entry in json_entry:
            bbox = entry["Geometry"]["BoundingBox"]
            x, y, w, h = bbox['Left'], bbox['Top'], bbox["Width"], bbox["Height"]
            bbox = [x, y, x + w, y + h]
            # bbox = normalize_box(bbox, width=1, height=1, size=resize_scale)
            ocrs[entry["Page"]].append({"word": entry["Text"], "bbox": bbox})

        ocr_path = {}
        for pg in set(pages):
            path_name = os.path.join(
                save_folder_ocr, f"{ocr_file_path.split('/')[-1].split('.')[0]}_{pg}.json")
            with open(path_name, "w") as f:
                json.dump(ocrs[pg], f)
            ocr_path[pg] = path_name

        return ocr_path

    except:
        return {}

# Function to get the images from the PDFs as well as the OCRs for the corresponding images without saving the image
def get_image_ocrs_dict_from_path(pdf_file_path: str, ocr_file_path: str):

    #try:
        # Getting the image list, since the pdfs can contain many image
        reader = PdfReader(pdf_file_path)
        img_list = {}
        pg_count = 1
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            for image_file_object in page.images:
                pdf_name = pdf_file_path.split('/')[-1].split('.')[0]
                pg_count += 1
                img_list[pg_count - 1] = pdf_file_path

        json_entry = json.load(open(ocr_file_path, 'r'))[1]
        json_entry = [x for x in json_entry["Blocks"] if "Text" in x]

        pages = [x["Page"] for x in json_entry]
        ocrs = {pg: [] for pg in set(pages)}

        for entry in json_entry:
            bbox = entry["Geometry"]["BoundingBox"]
            x, y, w, h = bbox['Left'], bbox['Top'], bbox["Width"], bbox["Height"]
            bbox = [x, y, x + w, y + h]
            ocrs[entry["Page"]].append({"word": entry["Text"], "bbox": bbox})

        ocr_path = {}
        for pg in set(pages):
            ocr_path[pg-1] = ocr_file_path
        
        #print(img_list)
        return img_list, ocr_path

#     except:
#         return {}, {}

