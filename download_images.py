#!/usr/bin/env python
# coding: utf-8

import bson
import requests
import os
import multiprocessing as mp
import re
from tqdm import tqdm
from bson import json_util
from natsort import natsorted  # Natural sorting

# Load Pinterest data
pins_bson = open('data/pinterest_iccv/subset_iccv_board_pins.bson', 'rb').read()
board_to_pins = {}
for obj in bson.decode_all(pins_bson):
    key = obj['board_id']
    val = obj['pins']
    board_to_pins[key] = val
print("Number of users: {}".format(len(board_to_pins)))

im_bson = open('data/pinterest_iccv/subset_iccv_pin_im.bson', 'rb').read()
pin_to_img = {}
for obj in bson.decode_all(im_bson):
    key = obj['pin_id']
    val = obj['im_url']
    pin_to_img[key] = val
print("Number of pins: {}".format(len(pin_to_img)))


def download_image(task_queue, progress_bar):
    '''
    Worker function to download images from a task queue.
    Updates the progress bar for each successful download.
    '''
    while not task_queue.empty():
        user_id, img_id, img_url = task_queue.get()
        try:
            img_obj = requests.get(img_url, allow_redirects=True)
            with open(f'data/pinterest_images/{img_id}.jpg', 'wb') as img_file:
                img_file.write(img_obj.content)
            progress_bar.update(1)  # Update the progress bar after each download
        except Exception as e:
            print(f"Failed to download image {img_id}: {e}")


if __name__ == "__main__":
    # Initialize variables
    img_to_id = {}
    board_to_userid = {}
    user_id = 0
    img_id_iter = 0
    task_queue = mp.Queue()

    # Prepare data for users-to-images mapping file
    filepath = 'data/users_to_images.train'
    if not os.path.exists("data/pinterest_images"):
        os.makedirs("data/pinterest_images")

    # Assign a unique `img_id` to each `img_url` in the order pins are processed
    for img_url in natsorted(pin_to_img.values()):
        if img_url not in img_to_id:
            img_to_id[img_url] = img_id_iter
            img_id_iter += 1

    # Populate the task queue with download tasks and write dataset file
    with open(filepath, "w") as f:
        # Use tqdm to track progress over all pins
        total_pins = sum(len(pins) for pins in board_to_pins.values())
        with tqdm(total=total_pins, desc="Processing Pins") as pbar:
            for board_id, pins in board_to_pins.items():
                # Find user_id corresponding to board_id
                if board_id in board_to_userid:
                    user_id = board_to_userid[board_id]
                else:
                    board_to_userid[board_id] = user_id
                    user_id += 1

                for pin in pins:
                    if pin in pin_to_img:
                        img_url = pin_to_img[pin]
                        img_id = img_to_id[img_url]  # Get the naturally sorted image ID

                        # Add the download task to the queue
                        task_queue.put((user_id, img_id, img_url))

                        # Write dataset entry to file
                        f.write(f"{user_id}\t{img_id}\t{img_url}\n")
                    
                    # Update tqdm progress bar for each processed pin
                    pbar.update(1)

    # Start multiprocessing pool for image downloading
    num_processes = mp.cpu_count()  # Number of parallel processes

    # Start a tqdm progress bar for image downloads
    with tqdm(total=task_queue.qsize(), desc="Downloading Images") as download_pbar:
        processes = []
        for _ in range(num_processes):
            # Pass the download_pbar to each process
            p = mp.Process(target=download_image, args=(task_queue, download_pbar))
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

    print("Image download completed.")
