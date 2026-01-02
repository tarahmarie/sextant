import itertools
import nltk
import os
import re
import sqlite3
import unicodedata
from multiprocessing import Pool, cpu_count

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm

from hapaxes_1tM import remove_tei_lines_from_text
from util import get_project_name, getListOfFiles

def ensure_nltk_data():
    resources = ['punkt_tab', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

ensure_nltk_data()

# Cache stopwords and lemmatizer (expensive to create repeatedly)
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


### Utility Functions
def remove_combining_characters(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))


def extract_author_name(xml_body):
    author_pattern = re.compile(r'<author>([^,]+)', re.IGNORECASE | re.DOTALL)
    match_author = author_pattern.search(xml_body)
    
    if match_author:
        author = match_author.group(1).strip()
        author = re.sub(r'\s*\([\s\d-]*\)', '', author)
    else:
        author = "Unknown Author"
    
    author = author.replace('-', '_')
    return author


def preprocess_text(text):
    """Preprocess a single text document."""
    text = remove_tei_lines_from_text(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s\u00C0-\u00FF]", "", text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
    return " ".join(tokens)


def process_single_file(file_path):
    """Worker function to process a single file."""
    with open(file_path, 'r') as f:
        body = f.read()

    author = remove_combining_characters(extract_author_name(body))
    title = file_path.split('/')[4].split('-')[1]
    text = preprocess_text(body)
    chapter_num = file_path.split('_')[-1]

    return (author, title, chapter_num, text)


def process_unseen_file(file_path):
    """Worker function to process a single unseen file."""
    with open(file_path, 'r') as f:
        body = f.read()
    return preprocess_text(body)


class AuthorshipAnalyzer:
    """Encapsulates the authorship analysis workflow."""
    
    def __init__(self, project_name):
        self.project_name = project_name
        self.db_path = f'./projects/{project_name}/db/svm.db'
        self.connection = None
        self.cursor = None
        
        # Data containers
        self.raw_data = []
        self.authors = []
        self.novels = []
        self.chap_nums = []
        self.chapters = []
        self.chapter_labels = []
        
        # ML components
        self.vectorizer = TfidfVectorizer()
        self.svm = SVC(kernel="linear")
        self.X = None
        
        # Multiprocessing settings
        self.num_workers = cpu_count()

    ### Database Methods
    def connect_db(self):
        """Create database connection."""
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

    def close_db_connection(self):
        """Close the SQLite database connection."""
        try:
            if self.connection:
                self.connection.close()
                print("Database connection closed.")
        except Exception as e:
            print(f"Error while closing the database connection: {str(e)}")

    def prepare_the_db(self):
        """Initialize database tables."""
        self.cursor.execute("DROP TABLE IF EXISTS chapter_assessments;")
        self.cursor.execute("DROP TABLE IF EXISTS test_set_preds;")
        self.cursor.execute("DROP TABLE IF EXISTS svm_coefficients;")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS chapter_assessments (novel TEXT, number TEXT);")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS test_set_preds (file TEXT);")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS svm_coefficients (feature_name TEXT, coefficient_value REAL);")
        self.connection.commit()

    def insert_chapter_data(self, data):
        """Batch insert chapter assessment data."""
        if not data:
            return
        num_columns = len(data[0])
        placeholders = ','.join(['?'] * num_columns)
        query = f"INSERT INTO chapter_assessments VALUES ({placeholders})"
        self.cursor.executemany(query, data)
        self.connection.commit()

    def insert_test_set_data(self, data):
        """Batch insert test set data."""
        if not data:
            return
        num_columns = len(data[0])
        placeholders = ','.join(['?'] * num_columns)
        query = f"INSERT INTO test_set_preds VALUES ({placeholders})"
        self.cursor.executemany(query, data)
        self.connection.commit()

    def update_the_chapters_table(self, column_names):
        """Add author columns to assessment tables."""
        unique_names = sorted(set(column_names))
        for name in unique_names:
            safe_name = name.replace(' ', '_')
            self.cursor.execute(f"ALTER TABLE chapter_assessments ADD COLUMN `{safe_name}` REAL;")
            self.cursor.execute(f"ALTER TABLE test_set_preds ADD COLUMN `{safe_name}` REAL;")
        self.connection.commit()

    def insert_coefficients_data(self, feature_names, coefficients):
        """Batch insert SVM coefficients."""
        try:
            data = list(zip(feature_names, coefficients))
            query = "INSERT INTO svm_coefficients (feature_name, coefficient_value) VALUES (?, ?)"
            self.cursor.executemany(query, data)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error occurred: {e}")

    ### Data Processing Methods
    def process_raw_files(self):
        """Process all raw files using multiprocessing."""
        all_files = getListOfFiles(f'./projects/{self.project_name}/splits')
        chunksize = max(1, len(all_files) // (self.num_workers * 4))

        with Pool(processes=self.num_workers) as pool:
            pbar = tqdm(
                desc='Loading and preprocessing files',
                total=len(all_files),
                colour="#00875f",
                bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]'
            )
            
            for result in pool.imap_unordered(process_single_file, all_files, chunksize=chunksize):
                self.raw_data.append(result)
                pbar.update(1)
            
            pbar.close()

    def build_lists(self):
        """Extract data into separate lists."""
        self.authors = [item[0] for item in self.raw_data]
        self.novels = [item[1] for item in self.raw_data]
        self.chap_nums = [item[2] for item in self.raw_data]
        self.chapters = [item[3] for item in self.raw_data]

    def prepare_labels(self):
        """Assign labels to chapters."""
        self.chapter_labels = list(self.novels)  # Use novels as labels

    def prepare_features(self):
        """Vectorize the chapter texts."""
        self.X = self.vectorizer.fit_transform(self.chapters)

    ### SVM Methods
    def test_model(self):
        """Test the SVM model and print accuracy."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.chapter_labels, test_size=0.30, random_state=42
        )

        self.svm.fit(X_train, y_train)

        y_pred = self.svm.predict(X_test)
        accuracy = self.svm.score(X_test, y_test)
        print("Accuracy:", accuracy)

        report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(report)

    def assess_authorship_likelihood(self):
        """Compute authorship likelihood for all chapters."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.novels, test_size=0.2, random_state=42, stratify=self.novels
        )
        self.svm.fit(X_train, y_train)

        # Vectorize all chapters at once (much faster than one at a time)
        all_vectorized = self.vectorizer.transform(self.chapters)
        all_scores = self.svm.decision_function(all_vectorized)

        # Normalize all scores at once
        scaler = MinMaxScaler()
        all_scores_normalized = scaler.fit_transform(all_scores)

        outcomes_dict = {}
        column_names = list(self.svm.classes_)

        pbar = tqdm(
            desc='Computing Authorship Likelihood',
            total=len(self.chapters),
            colour="#FB3FA8",
            bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]'
        )

        for i, (author, novel, chapter, chap_num) in enumerate(
            zip(self.authors, self.novels, self.chapters, self.chap_nums)
        ):
            outcome = {novel: score for novel, score in zip(self.svm.classes_, all_scores_normalized[i])}
            outcomes_dict[f"{novel}-{chap_num}"] = outcome
            pbar.update(1)

        pbar.close()
        return outcomes_dict, column_names

    def prepare_chapter_data(self, column_names, outcomes_dict):
        """Prepare and insert chapter assessment data."""
        chapter_transactions = []
        sorted_columns = sorted(set(column_names))

        pbar = tqdm(
            desc='Preparing Chapter Data',
            total=len(outcomes_dict),
            colour="#a361f3",
            bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]'
        )

        for key, value in outcomes_dict.items():
            novel = key.split('-')[0]
            novel = unicodedata.normalize('NFKD', novel)
            chap_num = key.split('-')[1]
            
            # Build tuple with scores in consistent column order
            scores = tuple(value.get(author, 0.0) for author in sorted_columns)
            chapter_transactions.append((novel, chap_num) + scores)
            pbar.update(1)

        pbar.close()
        
        self.update_the_chapters_table(column_names)
        self.insert_chapter_data(chapter_transactions)

    def unseen_test(self):
        """Test the model on unseen files using multiprocessing."""
        unseen_files = getListOfFiles(f'./projects/{self.project_name}/splits')
        chunksize = max(1, len(unseen_files) // (self.num_workers * 4))

        # Process files in parallel
        unseen_chapters = []
        with Pool(processes=self.num_workers) as pool:
            pbar = tqdm(
                desc='Preprocessing unseen files',
                total=len(unseen_files),
                colour="#00afff",
                bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]'
            )
            
            for result in pool.imap(process_unseen_file, unseen_files, chunksize=chunksize):
                unseen_chapters.append(result)
                pbar.update(1)
            
            pbar.close()

        # Train on split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.chapter_labels, test_size=0.2, random_state=42, stratify=self.chapter_labels
        )
        self.svm.fit(X_train, y_train)

        # Batch transform and predict (much faster)
        unseen_features = self.vectorizer.transform(unseen_chapters)
        unseen_predictions = self.svm.predict(unseen_features)
        confidence_scores = self.svm.decision_function(unseen_features)

        # Normalize all confidence scores at once
        scaler = MinMaxScaler()
        confidence_scores = scaler.fit_transform(confidence_scores)

        # Statistics
        print("\n")
        print("Range of scores:", np.min(confidence_scores), "to", np.max(confidence_scores))
        print("Mean score:", np.mean(confidence_scores))
        print("Standard deviation of scores:", np.std(confidence_scores))
        print("\n")

        # Build transactions
        test_transactions = []
        sorted_classes = list(self.svm.classes_)

        pbar = tqdm(
            desc='Building test results',
            total=len(unseen_chapters),
            colour="#ff6666",
            bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]'
        )

        for i, file_path in enumerate(unseen_files):
            better_filename = file_path.split('/')[5]
            scores = tuple(confidence_scores[i])
            test_transactions.append((better_filename,) + scores)
            pbar.update(1)

        pbar.close()
        self.insert_test_set_data(test_transactions)

    def track_model_coefficients(self):
        """Save SVM feature coefficients to database."""
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.svm.coef_.toarray().flatten()
        self.insert_coefficients_data(feature_names, coefficients)

    ### Main Workflow
    def build_the_thing(self):
        """Run the complete analysis workflow."""
        print("Preparing the db...")
        self.prepare_the_db()

        print("\nTesting the model before proceeding...\n")
        self.test_model()

        outcomes_dict, column_names = self.assess_authorship_likelihood()
        self.prepare_chapter_data(column_names, outcomes_dict)

        print("\nNow, testing the unseen texts...\n")
        self.unseen_test()

        self.track_model_coefficients()


def make_directories_if_needed_and_warn(project_name):
    """Ensure required directories exist."""
    exit_when_complete = False
    testset_path = f'./projects/{project_name}/testset'
    
    if not os.path.exists(testset_path):
        os.makedirs(testset_path)
        exit_when_complete = True
        print("\nI've just created the directory 'testset'. Make sure it has unseen/target texts before running again!")
    
    if exit_when_complete:
        exit()


def main():
    project_name = get_project_name()
    
    # Ensure directories exist
    make_directories_if_needed_and_warn(project_name)

    # Initialize analyzer
    analyzer = AuthorshipAnalyzer(project_name)
    analyzer.connect_db()

    try:
        print("\nLoading raw files...\n")
        analyzer.process_raw_files()
        analyzer.build_lists()
        analyzer.prepare_labels()

        unique_labels = np.unique(analyzer.authors)
        print(f"Found {len(unique_labels)} unique authors")

        print("\nVectorizing texts...")
        analyzer.prepare_features()

        analyzer.build_the_thing()
        print(f"Vocabulary size: {analyzer.X.shape[1]}")

    finally:
        analyzer.close_db_connection()


if __name__ == "__main__":
    main()