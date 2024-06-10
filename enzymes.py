import config
import json

with open(config.brenda_json) as brenda_json:
    brenda = json.load(brenda_json)

enzymes = list(brenda['data'].keys())

# exclude "spontaneous reaction"
enzymes = enzymes[1:]
