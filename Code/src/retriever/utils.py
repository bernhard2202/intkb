import re
import string
import unicodedata
import os


def get_filename_for_article_id(wiki_id):
    def _get_folder_for_id(_id):
        first = _id[0]
        if not first.isalpha() and not first.isdigit():
            return 'other'
        return first
    wiki_id = wiki_id.replace('/', '_slash_')
    return os.path.join(_get_folder_for_id(wiki_id), '{}.txt'.format(wiki_id))

