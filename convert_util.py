import fileinput
import heapq
import os
import sys
from optparse import OptionParser


def convert_to_dense(filename, last_feature=700):
    for line in fileinput.FileInput(filename, inplace=1):
        data = line.split()

        query_header = " ".join(data[:2])
        values = data[2:]
        prior_feature = 0
        features = ""

        for value in values:
            feature_info = value.split(':')
            feature = int(feature_info[0])
            while prior_feature + 1 != feature:
                features += str(prior_feature + 1) + ':0.0 '
                prior_feature += 1

            features += str(value) + ' '
            prior_feature += 1

        for i in range(prior_feature, last_feature):
            features += str(i) + ':0.0 '

        print(query_header + ' ' + features)


def sort_list(filename, output_file):
    last_query = None
    scores = []
    open(output_file, "w")
    with open(filename) as f:
        lines = f.readlines()
        line_count = len(lines)
        print(line_count)
        count = 0
        for line in lines:
            value = line.split()
            query, document, score = value[0], value[1], value[2]
            if last_query and last_query != query or count == line_count - 1:
                output = open(output_file, "a")
                while len(scores) > 0:
                    score = -heapq.heappop(scores)
                    output_line = str(score) + "\n"
                    output.write(output_line)
                output.close()
                last_query = query
            else:
                if not last_query:
                    last_query = query
                heapq.heappush(scores, -float(score))
            count += 1


def convert_to_predict_file(filename, output_file):
    with open(filename) as f:
        lines = f.readlines()
        line_count = len(lines)
        count = 0
        with open(output_file, "w") as w:
            for line in lines:
                value = line.split()
                query, document, score = value[0], value[1], value[2]
                if count == line_count - 1:
                    w.write(str(score))
                else:
                    w.write(str(score) + "\n")
                count += 1


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-c", "--convert_to_dense",
                      help="Converts sparse dataset to dense")
    parser.add_option("-s", "--sort_list",
                      help="Converts Ranklist to sorted documents")
    parser.add_option("-o", "--output",
                      help="File to output result")
    parser.add_option("-p", "--predict_list_input", )

    (options, args) = parser.parse_args()
    print(options)

    if options.convert_to_dense:
        if os.path.isabs(options.convert_to_dense):
            convert_to_dense(options.convert_to_dense)
        elif options.convert_to_dense[0] == '/':
            convert_to_dense(os.getcwd() + options.convert_to_dense)
        else:
            convert_to_dense(os.getcwd() + '/' + options.convert_to_dense)

    if options.sort_list:
        sort_list(options.sort_list, options.output)
    elif options.predict_list_input:
        convert_to_predict_file(options.predict_list_input, options.output)
