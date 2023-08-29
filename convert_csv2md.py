import argparse
from csv2md.table import Table

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="a script to do stuff")
    parser.add_argument("-i")
    parser.add_argument("-o")
    args=parser.parse_args()

    with open(str(args.i),'r') as f:
        table = Table.parse_csv(f)
    with open(str(args.o),'w') as f:
        f.write(table.markdown())

    print(table.markdown())
