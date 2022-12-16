"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Download COCO dataset images from a txt file
"""

import requests
import time
import concurrent.futures
import argparse
from os.path import exists
from os import makedirs

def download_image(img_url, dataset):
    """
    Download image from URL
    """
    img_bytes = requests.get(img_url).content
    img_name = img_url.split('/')[-1]

    # Save the image on ../../data/train/ ou ../../data/val/
    with open(f'../../data/{dataset}/' + img_name, 'wb') as img_file:
        img_file.write(img_bytes)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download COCO dataset.')
    parser.add_argument('-i', '--input',
                        help='Path to the txt file of URLs')
    parser.add_argument('-d', '--dataset',
                        help='Type of dataset \'train\' or \'val\'')

    args = parser.parse_args()
    input_file = args.input
    dataset = args.dataset

    if not exists(f'../../data/{dataset}'):
        makedirs(f'../../data/{dataset}')

    # Read all URLs and exclude the \n char
    with open(input_file, 'r') as url_list:
        img_urls = [str(line[0:-1]) for line in url_list.readlines()]
    print('Images to download:', len(img_urls), flush=True)
    
    t1 = time.perf_counter()

    # Download the images
    with concurrent.futures.ThreadPoolExecutor() as executor:
        print('Downloading images...')
        for url in img_urls:
            executor.submit(download_image, url, dataset)

    t2 = time.perf_counter()

    print(f'Finished in {t2-t1} seconds')