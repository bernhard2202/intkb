""" script that downloads all Wikipedia articles necessary for kb-completion and stores them to files """
import json
import argparse
import os
from tqdm import tqdm
import glob
import requests
import logging
from multiprocessing import Pool as ProcessPool
import urllib.parse as uparse

from retriever.utils import get_filename_for_article_id


def _get_folder_for_id(id):
    first = id[0]
    if not first.isalpha() and not first.isdigit():
        return 'other'
    return first


def aggregate_wiki_ids():
    article_ids = []
    stored = {}
    processed = 0

    for filename in glob.glob(os.path.join(args.input_data, '*.json')):
        logger.info('Reading "{}"..'.format(filename))
        with open(filename, 'r') as f:
            jdata = json.load(f)
            for query in jdata:
                name = query['wikipedia_link'].split('wiki/')[-1]
                if name not in stored:
                    article_ids.append(name)
                    stored[name] = 0
            processed += len(jdata)
    del stored
    logger.info('Processed {} datapoints, over {} unique wikipedia articles'.format(processed, len(article_ids)))
    return article_ids


def fetch_full_text(title):
    url = "https://en.wikipedia.org/w/api.php"
    querystring = {"action": "query",
                   "format": "json",
                   "titles": uparse.unquote(title),
                   "prop": "extracts",
                   "explaintext": "",
                   "exlimit": "max",
                   "redirects": ""}
    try:
        response = requests.request("GET", url, params=querystring, timeout=15)
    except requests.exceptions.ReadTimeout:
        response = requests.request("GET", url, params=querystring, timeout=60)
    json_response = json.loads(response.text).get('query').get('pages')
    key = list(json_response.keys())
    return json.loads(response.text).get('query').get('pages').get(key[0]).get('extract')


def download_wiki(ids):
    workers = ProcessPool(10)
    failed = 0
    skipped = 0
    success = 0
    with tqdm(total=len(ids)) as pbar:
        for result_code in tqdm(workers.imap_unordered(download_article, ids)):
            if result_code < 0:
                failed += 1
            elif result_code == 0:
                skipped += 1
            elif result_code == 1:
                success += 1
            else:
                assert False, 'unknown result code'
            pbar.update()
    workers.close()
    logger.info('Downloaded {} articles! {} articles not found, {} skipped (cached).'.format(success, failed, skipped))


def download_article(wiki_id):
    if wiki_id.startswith('Category:'):
        logger.info('Skip category..')
        return 0
    if len(wiki_id) == 0:
        logger.warning('Empty id..')
        return -1
    first = _get_folder_for_id(wiki_id)

    if not os.path.exists(os.path.join(args.save_path, first)):
        try:
            os.mkdir(os.path.join(args.save_path, first))
        except OSError:
            logger.fatal("Creation of the directory %s failed" % os.path.join(args.save_path, first))
            exit(-1)
        else:
            logger.info("Creating directory %s" % os.path.join(args.save_path, first))

    if os.path.exists(os.path.join(args.save_path, get_filename_for_article_id(wiki_id))):
        logger.info('Skip cached file {}'.format(wiki_id))
        return 0
    text = fetch_full_text(wiki_id)
    if text is None or len(text) < 2:
        logger.warning('Error with id {}'.format(wiki_id))
        return -1
    with open(os.path.join(args.save_path, get_filename_for_article_id(wiki_id)), 'w') as f:
        f.write(text)
    return 1


def setup():
    ids = aggregate_wiki_ids()
    ids = ids
    logger.info('Start download script..')
    download_wiki(ids)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', type=str, help='/path/to/input/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/wikipedia/data')
    args = parser.parse_args()

    setup()
