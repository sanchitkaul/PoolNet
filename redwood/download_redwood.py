#open json files
import os
import json
import requests
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup

def main0():
    base_url = "http://redwood-data.org/3dscan/dataset.html?c=car&i=278"
    base_url = "http://redwood-data.org/3dscan/dataset.html"
    base_url = "http://redwood-data.org/3dscan/data/video/"

    dest_data_path = Path('/media/SharedStorage/redwood')
    
    categories_path = Path("./categories.json")
    try:
        with open(categories_path, 'r') as f:
            categories = json.load(f)
    except Exception as e:
        print(f"Error loading categories.json: {e}")

    if categories is None:
        print("No categories found in provided path. Please check file path.")
        return
    else:
        for category, scans in categories.items():
            for scan in scans:
                print(f"Loading category: {category}, scans: {len(scan)}")
                obj_url = f"?c={category}&i={int(scan)}"

                # Get downlaod link
                response = requests.get(base_url + obj_url)
                if response.status_code == 200:
                    print(f"Successfully fetched data for {category} scan {scan}")
                    soup = BeautifulSoup(response.text, 'html.parser')
                    print(soup.prettify())
                    # print prettify to html file
                    with open(f"{category}_scan_{scan}.html", 'w') as f:
                        f.write(soup.prettify())
                    a_tag = soup.find('a', class_='grey waves-effect waves-light btn')
                    
                    print(a_tag)
                    print(a_tag.has_attr('href'))
                    if a_tag and a_tag.has_attr('href'):
                        download_link = a_tag['href']
                        print(f"Download link for {category} scan {scan}: {download_link}")

                        # download file
                        download_response = requests.get(download_link, stream=True)
                        if download_response.status_code == 200:
                            file_name = dest_data_path / f"{scan}.zip"
                            with open(file_name, 'wb') as f:
                                for chunk in tqdm(download_response.iter_content(chunk_size=8192), desc=f"Downloading {file_name}", unit='B', unit_scale=True):
                                    f.write(chunk)
                            print(f"Downloaded {file_name} successfully.")
                        else:
                            print(f"Failed to download {category} scan {scan}, status code: {download_response.status_code}")
                        
                else:
                    print(f"Failed to fetch data for {category} scan {scan}, status code: {response.status_code}")

                return
        
def main():
    # get all rgbd IDs
    rgbds_path = Path("./rgbds.json")
    progress_path = Path("./progress.json")
    dest_data_path = Path('/media/SharedStorage/redwood/mp4')

    base_url = "http://redwood-data.org/3dscan/data/video/"
    
    with open(rgbds_path, 'r') as f:
        rgbds = json.load(f)
    
    if rgbds is None:
        print("No RGBD IDs found in provided path. Please check file path.")
        return
    else:
        try:
            with open(progress_path, 'r') as f:
                progress = json.load(f)
            print(f"Loaded progress: {progress}")
        except FileNotFoundError:
            progress = {}
            print("No progress file found. Starting from scratch.")
        
        for rgbd_id in tqdm(rgbds):
            # download mp4 files
            if rgbd_id in progress:
                print(f"Skipping {rgbd_id}, already downloaded.")
                continue
            print(f"Downloading RGBD ID: {rgbd_id}")
            video_url = base_url + f"{rgbd_id}.mp4"

            response = requests.get(video_url, stream=True)
            with open(dest_data_path / f"{rgbd_id}.mp4", 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading {rgbd_id}.mp4", unit='B', unit_scale=True):
                    f.write(chunk)

            print(f"Downloaded {rgbd_id}.mp4 successfully.")
            progress[rgbd_id] = "downloaded"
            with open(progress_path, 'w') as f:
                json.dump(progress, f, indent=4)

            

if __name__ == "__main__":
    main()