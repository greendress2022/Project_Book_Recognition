import json
import os
import sys

# Load database from JSON file
base_dir = os.getcwd()
# Combine the base directory with the relative path
file_path = os.path.join(base_dir, "check_book_existence/database/fixedDatabase.json")
with open(file_path, "r") as file:
    database = json.load(file)
    #print(f'database accessed successfully, total rows: {len(database)}')

def book_exists(database, isbn10, isbn13):
    # Ensure isbn10 and isbn13 are strings or None
    if not (isinstance(isbn10, str) or isbn10 is None) or not (isinstance(isbn13, str) or isbn13 is None):
        raise TypeError("isbn10 and isbn13 must be strings or None")

    for book in database:
        # Check if either isbn10 or isbn13 matches in the book
        if isbn10 and isbn10 in book.get("isbn10", []):
            return True
        if isbn13 and isbn13 in book.get("isbn13", []):
            return True
    
    return False


# # Example tests

# print(book_exists(database, {"isbn13": "9780889955707"}))  # True
# print(book_exists(database, {"isbn10": "1419708813"}))  # True
# print(book_exists(database, {"isbn10": "1419708800"}))  # False
# print(book_exists(database, {"title": "qaqavii", "author": "Miriam Korner"}))  # True
# print(book_exists(database, {"title": "Qaqaviv", "author": "Miriam Korner"}))  # false
# print(book_exists(database, {"title": "Qaqavii", "author": "Miriam Korner", "pubDate": "2000"}))  # False, "pubDate":""
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Insufficient arguments", file=sys.stderr)
        sys.exit(1)

    filter1 = sys.argv[1]
    filter2 = sys.argv[2]
    print(f"The book with ISBN10: {filter1} ISBN13: {filter2} has been searched in the database.")
    result =  book_exists(database=database, isbn10= filter1, isbn13=filter2)
    print(f'Does this book exist in the database? => {result}')
