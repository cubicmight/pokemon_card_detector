import os
from os.path import basename, exists
import requests
import shutil
import progressbar

from bs4 import BeautifulSoup

card_url = 'https://pkmncards.com/page/%s/?s=set:crimson-invasion&display=images&sort=date&order=asc'

# store all pokemon card url
pokemon_entry_urls = []

# there is only one page for crimson invasion, some sets have more
for i in range(1, 2):
    # build full url
    card_url_full = card_url % str(i)

    # http request to get page
    r = requests.get(card_url_full)
    # get text from request
    page_html = r.text
    # parse the request
    page_soup = BeautifulSoup(page_html, "html.parser")
    # find tag which has link to image file
    entries = page_soup.find_all("div", {"class": "entry-content"})
    # collect all links in a list
    for entry in entries:
        pokemon_entry_urls.append(entry.a['href'])

# we will store all image here
folder_path = './all_pokemon'
# create if it does not exist
if not exists(folder_path):
    os.mkdir(folder_path)

# widgets = ['Loading: ', progressbar.AnimatedMarker()]
widgets = [' [',
     progressbar.Timer(format='elapsed time: %(elapsed)s'),
     '] ',
     progressbar.Bar('*'), ' (',
     progressbar.ETA(), ') ',
     ]

bar = progressbar.ProgressBar(max_value=len(pokemon_entry_urls), widgets=widgets).start()

# go over the list and download + store all card images
i = 0;
for entry_url in pokemon_entry_urls:
    i = i + 1
    bar.update(i)
    # start the http request
    r = requests.get(entry_url)
    # get text
    page_html = r.text
    # parse it
    page_soup = BeautifulSoup(page_html, "html.parser")
    # find image tag
    card_img_url = page_soup.find("div", {"class": "card-image-area"}).a['href']

    # find name from the tag
    img_name_from_url = basename(card_img_url)
    # clean up name
    file_name = img_name_from_url[10:]
    file_name = file_name[:file_name.index('?')]

    # where to save the file
    file_path = os.path.join(folder_path, file_name)
    # dont down load if it exists already
    if os.path.exists(file_path):
        continue

    # request to get image, it may be big so we stream if needed
    r = requests.get(card_img_url, stream=True)
    # is success we save it
    if r.status_code == 200:
        with open(file_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        print ("Error downloading URL: %s" % card_img_url)

