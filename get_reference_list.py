import os
import re
import requests

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

import config
import enzymes


with open(config.species_list) as species_list:
    bacteria = set(s.strip() for s in species_list.readlines())

for enzyme in tqdm(enzymes.enzymes):
    references = []

    url = f'https://www.brenda-enzymes.org/enzyme.php?ecno={enzyme}&onlyTable=Reference'

    references_html = BeautifulSoup(requests.get(url).content, features='html5lib')

    for row in references_html.find_all(id=re.compile('tab30r\d+sr0$')):
        fields = row.find_all(id=re.compile('tab30r\d+sr0c\d+$'))

        for organism in fields[7].get_text().split(','):
            organism = organism.strip()
            if organism in bacteria:
                references.append({
                    'enzyme': enzyme,
                    'organism': organism,
                    'reference': fields[0].get_text().strip(),
                    'pubmed_id': fields[8].get_text().strip()
                })

    references = pd.DataFrame(references)
    references.to_csv(
        config.references_file,
        mode='a',
        header = not os.path.exists(config.references_file)
    )
