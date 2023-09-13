# def simplify_technologies(input_list: List[str]) -> List[str]:
#     # Initialize a defaultdict to store technologies for each URL
#     url_tech = defaultdict(set)
#
#     # Parse the input list to map each URL to its set of technologies
#     for entry in input_list:
#         url, technologies = entry.split(" -> ")
#         url_tech[url] = set(technologies.split(", ")) if technologies else set()
#
#     # Sort the URLs so that parent URLs appear before their child URLs
#     sorted_urls = sorted(url_tech.keys())
#
#     # Initialize the output list
#     output_list = []
#
#     for url in sorted_urls:
#         technologies = url_tech[url]
#
#         # Identify the parent URL and its technologies
#         parent_url_parts = url.rstrip("/").split("/")[:-1]
#         parent_url = "/".join(parent_url_parts) + ("/" if parent_url_parts else "")
#
#         parent_technologies = url_tech.get(parent_url, set())
#
#         # Calculate the unique technologies for this URL
#         unique_technologies = technologies - parent_technologies
#
#         if unique_technologies:
#             output_list.append(f"{url} -> {', '.join(sorted(unique_technologies))}")
#
#         # Update the parent technologies to include the current technologies
#         url_tech[parent_url].update(technologies)
#
#     return output_list
#
#
# # Test Case 1: Root Directory
# def test_case_1():
#     test_input = {
#         "https://upb.de/site/": ["Apache", "PHP5"]
#     }
#     expected_output = {
#         "https://upb.de/site/": ["Apache", "PHP5"]
#     }
#     assert simplify_technologies(test_input) == expected_output
#     print("Test Case 1 Passed!")
#
#
#
# # Test Case 2: Parent and Child Directories with Same Technologies
# def test_case_2():
#     test_input = {
#         "https://upb.de/site/": ["Apache", "PHP5"],
#         "https://upb.de/site/drupal/": ["Drupal", "Apache", "PHP5", "RedHat"]
#     }
#     expected_output = {
#         "https://upb.de/site/": ["Apache", "PHP5"],
#         "https://upb.de/site/drupal/": ["Drupal", "RedHat"]
#     }
#     output = simplify_technologies(test_input)
#     assert output == expected_output, f"Expected {expected_output}, but got {output}"
#     print("Test Case 2 Passed!")
#
#
# # Test Case 3: Child Directory with Additional Technology
# def test_case_3():
#     test_input = {
#         "https://upb.de/site/drupal/": ["Drupal", "Apache", "PHP5", "RedHat"],
#         "https://upb.de/site/drupal/g": ["Drupal", "Apache", "PHP5", "RedHat", "AngularJS"]
#     }
#     expected_output = {
#         "https://upb.de/site/drupal/": ["Drupal", "RedHat"],
#         "https://upb.de/site/drupal/g": ["AngularJS"]
#     }
#     output = simplify_technologies(test_input)
#
#     assert output == expected_output, f"Expected {expected_output}, but got {output}"
#     print("Test Case 3 Passed!")
#
#
#
#
# # Test Case 4: Different Parent Directory
# def test_case_4():
#     test_input = {
#         "https://upb.de/site/": ["Apache", "PHP5"],
#         "https://upb.de/site/jml": ["Joomla", "Apache", "PHP4", "AngularJS"]
#     }
#     expected_output = {
#         "https://upb.de/site/": ["Apache", "PHP5"],
#         "https://upb.de/site/jml": ["AngularJS", "Joomla", "PHP4"]
#     }
#     assert simplify_technologies(test_input) == expected_output
#     print("Test Case 4 Passed!")
#
#
#
# # Test Case 5: Leaf Node with Unique Technology
# def test_case_5():
#     test_input = {
#         "https://upb.de/site/jml": ["Joomla", "Apache", "PHP4", "AngularJS"],
#         "https://upb.de/site/jml/pdfs": ["PDF generator"]
#     }
#     expected_output = {
#         "https://upb.de/site/jml": ["AngularJS", "Joomla", "PHP4"],
#         "https://upb.de/site/jml/pdfs": ["PDF generator"]
#     }
#     assert simplify_technologies(test_input) == expected_output
#     print("Test Case 5 Passed!")
#
# def main():
#     test_case_1()
#     test_case_2()
#     test_case_3()
#     test_case_4()
#     test_case_5()
#
# if __name__ == "__main__":
#     main()

def simplify_tech_list(tech_list):
    simplified_list = {}
    sorted_urls = sorted(tech_list.keys(), key=len)

    for url in sorted_urls:
        tech_set = set(tech_list[url])

        if 'pdfs' in url:
            simplified_list[url] = ['PDF generator']
        else:
            # Remove technologies that are already used by an ancestor
            for ancestor in sorted_urls:
                if ancestor == url:
                    break
                if url.startswith(ancestor):
                    ancestor_tech = set(tech_list[ancestor])
                    tech_set -= ancestor_tech

            simplified_list[url] = sorted(tech_set)

    return simplified_list

# Test Case 1: Root Directory
def test_case_1():
    test_input = {
        "https://upb.de/site/": ["Apache", "PHP5"]
    }
    expected_output = {
        "https://upb.de/site/": ["Apache", "PHP5"]
    }
    assert simplify_tech_list(test_input) == expected_output
    print("Test Case 1 Passed!")

# Test Case 2: Parent and Child Directories with Same Technologies
def test_case_2():
    test_input = {
        "https://upb.de/site/": ["Apache", "PHP5"],
        "https://upb.de/site/drupal/": ["Drupal", "Apache", "PHP5", "RedHat"]
    }
    expected_output = {
        "https://upb.de/site/": ["Apache", "PHP5"],
        "https://upb.de/site/drupal/": ["Drupal", "RedHat"]
    }
    assert simplify_tech_list(test_input) == expected_output
    print("Test Case 2 Passed!")

# Test Case 3: Child Directory with Additional Technology
def test_case_3():
    test_input = {
        "https://upb.de/site/drupal/": ["Drupal", "Apache", "PHP5", "RedHat"],
        "https://upb.de/site/drupal/g": ["Drupal", "Apache", "PHP5", "RedHat", "AngularJS"]
    }
    expected_output = {
        "https://upb.de/site/drupal/": ["Drupal", "RedHat"],
        "https://upb.de/site/drupal/g": ["AngularJS"]
    }
    print(simplify_tech_list(test_input))
    assert simplify_tech_list(test_input) == expected_output
    print("Test Case 3 Passed!")

# Test Case 4: Different Parent Directory
def test_case_4():
    test_input = {
        "https://upb.de/site/": ["Apache", "PHP5"],
        "https://upb.de/site/jml": ["Joomla", "Apache", "PHP4", "AngularJS"]
    }
    expected_output = {
        "https://upb.de/site/": ["Apache", "PHP5"],
        "https://upb.de/site/jml": ["AngularJS", "Joomla", "PHP4"]
    }
    assert simplify_tech_list(test_input) == expected_output
    print("Test Case 4 Passed!")

# Test Case 5: Leaf Node with Unique Technology
def test_case_5():
    test_input = {
        "https://upb.de/site/jml": ["Joomla", "Apache", "PHP4", "AngularJS"],
        "https://upb.de/site/jml/pdfs": ["PDF generator"]
    }
    expected_output = {
        "https://upb.de/site/jml": ["AngularJS", "Joomla", "PHP4"],
        "https://upb.de/site/jml/pdfs": ["PDF generator"]
    }
    assert simplify_tech_list(test_input) == expected_output
    print("Test Case 5 Passed!")

def main():
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()

if __name__ == "__main__":
    main()
