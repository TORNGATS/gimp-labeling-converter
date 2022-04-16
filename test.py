
import argparse
import json
import logging
import os
import sys

from gimp_labeling_converter import list_handlers, phmError, run_translator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler(sys.stdout)],
)

parser = argparse.ArgumentParser(description="Processing images labeled by GIMP")
parser.add_argument('--dir_in', '--dir', '-d', type=str, help="Directory containing the images.")
parser.add_argument('--file_out', '--out', '-o', type=str, help="Output Directory")
parser.add_argument('--handler', '--type', required=False, default='mask', choices=list_handlers(), help="Handler Type")
parser.add_argument('--num_worker', '-w', type=int, required=False, default=1, help="Number of workers.")
parser.add_argument('--config', type=str, required=False, help='Configuration file path')
parser.add_argument('--name', '-n', type=str, required=False, default='', help='Dataset name')
parser.add_argument('--info', type=str, required=False, default='', help='Description')
parser.add_argument('--url', type=str, required=False, default='', help='URL')
parser.add_argument('--version', type=str, required=False, default='', help='Version')
parser.add_argument('--year', type=int, required=False, default=2022, help='Year')
parser.add_argument('--contrib', type=str, required=False, default='Parham Nooralishahi', help='Contributor')
parser.add_argument('--category', '-c', action="extend", nargs="+", type=str, help='Class Categories')

def main():

    args = parser.parse_args()
    parser.print_help()

    config = {}
    if args.config is not None:
        if not os.path.isfile(args.config) or not args.config.endswith('.json'):
            logging.error(f'{args.config} is not valid!')
            return
        
        with open(args.config, 'r') as fin:
            config = json.load(fin)
    else:
        config = vars(args)
        if args.dir_in is None or not os.path.isdir(args.dir_in):
            logging.error(f'{args.dir_in} is not a directory!')
            return
    
        if args.category is not None and len(args.category) != 0:
            cs = {}
            for i in range(len(args.category)):
                cs[args.category[i]] = (i + 1)
            config['category'] = cs
            print(cs)
            # Save the category lookup
            with open(os.path.join(args.file_out, 'categories.txt'), 'w') as fout:
                for (cname, cid) in cs.items():
                    fout.write(f'{cname},{cid}\n')

    run_translator(
        config['dir_in'], 
        config['handler'], **config)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except phmError as ex:
        logging.error(ex.message)