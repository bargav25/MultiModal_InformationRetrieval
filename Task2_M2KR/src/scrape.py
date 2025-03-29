import os
import time
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from config import PASSAGE_IMAGE_FOLDER, PASSAGE_SCRAPED_FOLDER
from PIL import UnidentifiedImageError, Image
from urllib.parse import unquote, quote


def download_image(url, folder_name, idx):

    if not url.startswith("https:"):
        url = "https:" + url

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/98.0.4758.102 Safari/537.36")
    }

    folder_path = os.path.join(PASSAGE_SCRAPED_FOLDER, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        file_path = os.path.join(folder_path, f"downloaded_image_{idx}.jpg")
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Saved: {file_path}")
    except (requests.RequestException, UnidentifiedImageError) as e:
        print(f"Failed to download {url}: {e}")


def get_wikipedia_url(img_filename):
    """Convert an image filename to a Wikipedia article URL."""
    base_name = os.path.splitext(img_filename)[0]
    decoded_name = unquote(base_name)  # 🔥 fix is here
    page_title = quote(decoded_name.replace(" ", "_"))  # re-encode properly
    return f"https://en.wikipedia.org/wiki/{page_title}"
    # page_title = quote(base_name.replace(" ", "_"))
    # return f"https://en.wikipedia.org/wiki/{page_title}"


def scrape_wikipedia_images(page_url):
    """Scrape valid image URLs from a Wikipedia page."""
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/98.0.4758.102 Safari/537.36")
    }

    try:
        response = requests.get(page_url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Page not found: {page_url} ({response.status_code})")
            return []
    except requests.RequestException as e:
        print(f"Error fetching {page_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.select('img')

    image_urls = [
        tag.get("src") for tag in img_tags
        if tag.get("src") and re.search(r'wikipedia/.*/thumb/', tag.get("src")) and not tag.get("src").endswith('.svg')
    ]
    return image_urls


def main():
    print("Loading image filenames from:", PASSAGE_IMAGE_FOLDER)
    image_names = os.listdir(PASSAGE_IMAGE_FOLDER)


    for img_filename in tqdm(image_names, desc="Scraping Wikipedia"):
        wiki_url = get_wikipedia_url(img_filename)
        print(f"\n Querying: {wiki_url}")

        image_links = scrape_wikipedia_images(wiki_url)
        print(f"Found {len(image_links)} image(s)")

        for idx, img_url in enumerate(image_links):
            folder_name = os.path.basename(wiki_url)
            download_image(img_url, folder_name, idx)

        time.sleep(1) # Be polite to Wikipedia servers


if __name__ == "__main__":
    main()