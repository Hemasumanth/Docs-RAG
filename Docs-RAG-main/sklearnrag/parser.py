from pathlib import Path
import html2text
from bs4 import BeautifulSoup
from sklearnrag.config import WORK_DIR

h = html2text.HTML2Text()

# Ignore converting links from HTML
h.ignore_links = False
h.mark_code = True
h.reference_links = True

def remove_examples_using_section(html_content):
    """
    Removes sections starting with 'examples-using' from the HTML content.

    :param html_content: HTML content as a string.
    :return: Modified HTML content as a string.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    sections_to_remove = soup.find_all("section", id=lambda x: x and x.startswith("examples-using"))
    for section in sections_to_remove:
        section.decompose()
    return str(soup)

def extract_text_from_section(section):
    """
    Converts an HTML section to markdown text using html2text.

    :param section: A BeautifulSoup section object.
    :return: Markdown text as a string.
    """
    response = h.handle(section.prettify())
    return response

def path_to_uri(path, scheme="https://", domain="scikit-learn.org/stable/"):
    """
    Converts a file path to a URI.

    :param path: Path to the file.
    :param scheme: URI scheme, default is 'https://'.
    :param domain: The domain name, default is 'scikit-learn.org/stable/'.
    :return: A URI as a string.
    """
    return scheme + domain + str(path).split(domain)[-1]

def extract_sections(record):
    """
    Extracts sections from an HTML file and converts them to markdown.

    :param record: A dictionary containing the path to the HTML file.
    :return: A list of dictionaries, each containing a source URI and the text of a section.
    """
    with open(record["path"], "r", encoding="utf-8") as html_file:
        html_content = remove_examples_using_section(html_file)
        soup = BeautifulSoup(html_content, "html.parser")

    sections = soup.find_all("section")
    section_list = []

    if len(sections) == 0:
        uri = path_to_uri(path=record["path"])
        section_text = h.handle(soup.prettify())
        if section_text:
            section_list.append({"source": f"{uri}", "text": section_text})
    else:
        for section in sections:
            section_id = section.get("id").strip()
            section_text = extract_text_from_section(section)
            if section_text:
                uri = path_to_uri(path=record["path"])
                section_list.append({"source": f"{uri}#{section_id}", "text": section_text})
    return section_list


def fetch_text(uri):
    """
    Fetches and returns the text from an HTML file based on the given URI.
    If an anchor is provided in the URI, it fetches text from the specific section.
    Otherwise, it returns the text from the entire HTML document.

    :param uri: The URI of the HTML document, optionally including an anchor (#).
    :return: The extracted text as a string.
    """
    url, anchor = uri.split("#") if "#" in uri else (uri, None)
    file_path = Path(WORK_DIR, url.split("https://")[-1])
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")
    if anchor:
        target_element = soup.find(id=anchor)
        if target_element:
            text = target_element.get_text()
        else:
            return fetch_text(uri=url)
    else:
        text = soup.get_text()
    return text