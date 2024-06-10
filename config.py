import configparser

config = configparser.ConfigParser()
config.read('config.ini')

brenda_json = config['data']['brenda_json']
references_file = config['data']['references_file']
species_list = config['data']['species_list']
