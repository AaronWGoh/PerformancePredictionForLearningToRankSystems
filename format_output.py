import os
from shutil import copyfile
from itertools import groupby


def sort_clusters(directory):
    different_evaluations = {"LISTWISE METRIC:	Precision",
                             "LISTWISE METRIC:	NDCG",
                             "LISTWISE METRIC:	RR",
                             "LISTWISE METRIC:	ERR"}
    for filename in os.listdir(directory):
        new_file = ""
        if filename[-3:] == '.txt':
            new_file = filename[:-3] + '_formatted.txt'
        copyfile(os.path.join(directory, filename), os.path.join(directory, new_file))
        with open(os.path.join(directory, new_file), 'w') as f:
            cluster_texts = {}
            cluster_num = 0
            in_cluster = False
            for line in f:
                if "\t0" in line:
                    cluster_num += 1
                    in_cluster = True
                if in_cluster:
                    print()

                for evaluation in different_evaluations:
                    if evaluation == line:
                        text.append(line)

                text.append(line.strip() +"\n")




if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--directory",
                      help="Location of output text files")