import wget
import argparse
import os
import zipfile


def geturl(line):
    return line.strip().split(" ")[-1]


def getname(url):
    return url.split("/")[3] + ".zip"

# extend with actually downloading monolingual opus data - opus get does not do that currently.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", help="file with ro opus zip locations as given by opus_get list")
    parser.add_argument("-o", "--output",
                        default="./raw/opus",
                        help="output dir where archives will be downloaded")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    archives = [
        wget.download(geturl(l),
                      out=os.path.join(args.output, getname(geturl(l))))
        for l in open(args.file).readlines()]

    for a in archives:
        with zipfile.ZipFile(a, 'r') as zip_ref:
            zip_ref.extractall(args.output)
        os.remove(a)
