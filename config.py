import tomllib

with open('config.toml', 'rb') as config:
    data = tomllib.load(config)
    brenda_json = data['brenda_json']
    references_file = data['references_file']
