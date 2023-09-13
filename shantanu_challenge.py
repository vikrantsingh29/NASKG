#
# def simplify_tech_list(tech_list):
#     simplified_list = {}
#     sorted_urls = sorted(tech_list.keys(), key=len)
#
#     for url in sorted_urls:
#         tech_set = set(tech_list[url].split(', '))
#
#         if 'pdfs' in url:
#             simplified_list[url] = 'PDF generator'
#         else:
#             # Remove technologies that are already used by an ancestor
#             for ancestor in sorted_urls:
#                 if ancestor == url:
#                     break
#                 if url.startswith(ancestor):
#                     ancestor_tech = set(tech_list[ancestor].split(', '))
#                     tech_set -= ancestor_tech
#
#             simplified_list[url] = ', '.join(sorted(tech_set))
#
#     return simplified_list
#
#
# def test_simplify_tech_list():
#     all_tests_passed = True
#     test_cases = [
#         {
#             'input': {
#                 'https://upb.de/site/': 'Apache, PHP5',
#                 'https://upb.de/site/drupal/': 'Drupal, Apache, PHP5, RedHat',
#                 'https://upb.de/site/drupal/g': 'Drupal, Apache, PHP5, RedHat, AngularJS',
#                 'https://upb.de/site/jml': 'Joomla, Apache, PHP4, AngularJS',
#                 'https://upb.de/site/jml/pdfs': 'PDF generator'
#
#             },
#             'output': {
#                 'https://upb.de/site/': 'Apache, PHP5',
#                 'https://upb.de/site/drupal/': 'Drupal, RedHat',
#                 'https://upb.de/site/drupal/g': 'AngularJS',
#                 'https://upb.de/site/jml': 'AngularJS, Joomla, PHP4',
#                 'https://upb.de/site/jml/pdfs': 'PDF generator'
#             }
#
#         },
#         # Additional test cases
#         {
#             'input': {
#                 'https://upb.de/site/': 'Apache, PHP5',
#                 'https://upb.de/site/a': 'Apache, PHP5',
#                 'https://upb.de/site/a/b': 'Apache, PHP5, RedHat'
#             },
#             'output': {
#                 'https://upb.de/site/': 'Apache, PHP5',
#                 'https://upb.de/site/a': '',
#                 'https://upb.de/site/a/b': 'RedHat'
#             }
#         },
#         {
#             'input': {
#                 'https://upb.de/site/': 'Apache',
#                 'https://upb.de/site/a': 'Apache, PHP5'
#             },
#             'output': {
#                 'https://upb.de/site/': 'Apache',
#                 'https://upb.de/site/a': 'PHP5'
#             }
#         },
#         {
#             'input': {
#                 'https://upb.de/site/': 'Apache',
#                 'https://upb.de/site/a': 'Apache'
#             },
#             'output': {
#                 'https://upb.de/site/': 'Apache',
#                 'https://upb.de/site/a': ''
#             }
#         },
#         {
#             'input': {
#                 'https://upb.de/site/': 'Apache, PHP5',
#                 'https://upb.de/site/a': 'Apache, PHP5, RedHat',
#                 'https://upb.de/site/a/b': 'Apache, PHP5, RedHat, AngularJS'
#             },
#             'output': {
#                 'https://upb.de/site/': 'Apache, PHP5',
#                 'https://upb.de/site/a': 'RedHat',
#                 'https://upb.de/site/a/b': 'AngularJS'
#             }
#         }
#     ]
#
#     for i, test_case in enumerate(test_cases):
#         result = simplify_tech_list(test_case['input'])
#         if result == test_case['output']:
#             print(f"Test case {i + 1} passed")
#         else:
#             print(f"Test case {i + 1} failed: got {result}, expected {test_case['output']}")
#             all_tests_passed = False
#
#     return all_tests_passed
#
# if __name__ == "__main__":
#     test_simplify_tech_list()

import os

# specify the directory where your text files are located
directory = 'C:\\Users\\vikrant.singh\\Desktop\\data_vikrant_new'

# specify the output file
output_file = 'C:\\Users\\vikrant.singh\\Desktop\\data_vikrant_new\\new_trajectory_data.txt'


# open the output file in write mode
with open(output_file, 'w') as outfile:
    # loop through each file in the directory
    for filename in os.listdir(directory):
        # only process files with a .txt extension
        if filename.endswith('.txt'):
            # open each file in read mode
            with open(os.path.join(directory, filename)) as infile:
                # write the contents of each file to the output file
                outfile.write(infile.read())
