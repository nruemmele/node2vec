"""
Create node2vec embeddings for yelp edgelists
"""
import fileinput
import os
from src import main as node2vec_main
from src import parse_args


def create_edgelist(filenames, outfilename):
    """
    Concatenate edgelists per each type of link into one big edgelist file
    :param filenames: list of filepaths where edge lists are
    :param outfilename: file path to write the final edge list
    :return:
    """
    with open(outfilename, 'w') as fout, fileinput.input(filenames) as fin:
        for line in fin:
            fout.write(line)


def node2vec_embed(input_file, out_file, p=1, q=1, workers=8):
    """
    Start node2vec
    :param input_file:
    :param out_file:
    :param p:
    :param q:
    :param workers:
    :return:
    """
    print("Starting node2vec embedding...")
    args = parse_args()
    args.input = input_file
    args.output = out_file
    args.p = p
    args.q = q
    args.workers = workers
    node2vec_main(args)


if __name__ == "__main__":

    dir_path = "/home/natalia/Projects/network-analysis/node2vec/graph/yelp"
    edgelists = [dir_path + "/yelp-sushi-review-business.edgelist/part-00000-88c5743b-1abd-4210-8b63-236de24e9077.csv",
                 dir_path + "/yelp-sushi-review-keyword.edgelist/part-00000-c4cb97ac-4312-4912-86e8-06d4b12649fc.csv",
                 dir_path + "/yelp-sushi-user-review.edgelist/part-00000-834d86d8-2885-495b-af05-270c2356e441.csv"]
    file_path = os.path.join(dir_path, "yelp-sushi.edgelist")
    create_edgelist(edgelists, file_path)

    node2vec_embed(input_file=file_path,
                   out_file="/home/natalia/Projects/network-analysis/node2vec/emb/yelp-sushi.emd")