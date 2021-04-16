"""
This file will parse and transform my
dataset into vectors.
"""

from collections import defaultdict
import argparse

import glob
import markdown
from tqdm import tqdm
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Parser:
    """
    Parse text into useable information.
    """
    def __init__(self, folder_loc):
        self.read_docs(folder_loc)
        self.htmlize_md()
        self.extract_info()

    def read_docs(self, folder_loc):
        """
        Read documents, keeping the markdown syntax.
        """
        names = glob.glob(f"{folder_loc}*.md")
        docs = []
        for name in names:
            curr_file = open(name)
            text = curr_file.read()
            docs.append((name, text))
            curr_file.close()
        self.raw_docs = docs

    def htmlize_md(self):
        """
        Convert each doc to HTML
        """
        converted = []
        for raw_doc in tqdm(self.raw_docs, desc="Converting to html..."):
            html = markdown.markdown(raw_doc[1])
            converted.append((raw_doc[0], html))

        self.html_docs = converted

    def extract_info(self):
        """
        Build a list of extracted information
        """

        parsed = []
        for doc in tqdm(self.html_docs, desc="extracting info..."):
            soup = BeautifulSoup(doc[1])
            
            data = defaultdict(list)
            data["file_name"] = (doc[0].split("/"))[-1]

            html_links = soup.find_all("a")
            data["file_links"] = [x.get("href") for x in html_links]
            data["link_text"] = "\n".join([x.getText() for x in html_links])
            data["title"] = soup.find("h1")
            data["paragraphs"] = "\n".join([x.getText() for x in
                                            soup.find_all("p")])

            parsed.append(data)

        return parsed


class Vectorizer:
    """
    Take each parsed element and convert it into a feature vector
    """
    def __init__(self, wiki_folder_path):
        self.parsed_data = Parser(wiki_folder_path).extract_info()
        self.make_unique_filelinks()
        self.make_text_vect()
        self.features = self.vectorize_all()

    def make_unique_filelinks(self):
        """
        Get a list of all the unique file links
        """

        self.unique_links = set()
        for doc_data in self.parsed_data:
            self.unique_links.update(doc_data["file_links"])
        self.num_links = len(self.unique_links)

    def make_text_vect(self):
        
        self.tfidf_vect = TfidfVectorizer(stop_words="english",
                                          ngram_range=(1, 2))

        corpus = []
        
        for x in self.parsed_data:
            corpus.append(x["paragraphs"])

        self.tfidf_vect.fit_transform(corpus)

    def vectorize_all(self):
        """
        Convert each document into a vector, based off of the features we have
        extracted.
        """
        
        output_file = open("featurized_text.tsv", "w")
        vecs = []
        file_output = []
        for doc_data in tqdm(self.parsed_data, desc="Vectorize all data points"):
            # each doc_data elemen is a dictionary
            vec = self.vectorize(doc_data)
            vecs.append(vec)
            if len(vec) != 0:

                line_text = "\t".join([str(x) for x in vec])
                line_text = doc_data["file_name"] + "\t" + line_text
                file_output.append(line_text)
        
        text = "\n".join(file_output)
        output_file.write(text)
        output_file.close()
        return vecs

    def vectorize(self, doc_data):
        """
        Convert one document into a vector.
        """
        # find all the unique file links
        file_links = self.filelinks_to_vec(doc_data["file_links"])
        para_vecs = self.vectorize_text(doc_data["paragraphs"])
        
        file_links.extend(para_vecs)
        return file_links

    def filelinks_to_vec(self, link_data):
        """
        Convert the filelinks to vector
        """

        vec = []
        link_set = list(set(link_data))
        for i, link in enumerate(self.unique_links):
            if link in link_set:
                vec.append(1)
            else:
                vec.append(0)
        return vec

    def vectorize_text(self, text_str):
        """
        Given some string, transform
        it into a vector with BERT.
        """
        result = self.tfidf_vect.transform([text_str])
        return list(result[0].toarray())[0]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("wiki_folder_location", help="Path to folder with your wiki files (must have .md extension), ex. /home/schar/Documents/wiki/")
    args = argparser.parse_args()
    
    vect = Vectorizer(args.wiki_folder_location)
