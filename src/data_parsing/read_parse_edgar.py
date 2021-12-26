import collections
import logging
import re
import xml.etree.ElementTree as ElementTree
from typing import Tuple, List, Dict


logger = logging.getLogger(__name__)

ENTITY_PREFIXES = ["us-gaap"]
ENTITY_FORMATS = ["ixt:numdotdecimal"]


def recursive_text_extract(
        et: ElementTree,
        entity_prefixes: List[str],
        entity_formats: List[str]
) -> List:
    text_parts = []
    for node in et:
        text_part = {
            "text": node.text if node.text else "",
            "sub": recursive_text_extract(node, entity_prefixes=entity_prefixes, entity_formats=entity_formats)
        }

        if any(entity_prefix == node.attrib.get("name", "") for entity_prefix in entity_prefixes) or \
                any(entity_format == node.attrib.get("format", "") for entity_format in entity_formats):
            text_part["entity"] = node.attrib
        else:
            text_part["entity"] = None
        text_part["tail"] = node.tail if node.tail else ""
        if text_part["text"] != "" or text_part["tail"] != "" or text_part["entity"] is not None:
            text_parts.append(text_part)
        elif len(text_part["sub"]) > 0:
            text_parts.append(text_part["sub"])
    return text_parts


def recursive_text_extract_flattened(
        et: ElementTree,
        entity_prefixes: List[str],
        entity_list: List[Dict],
        text: str = ""
) -> Tuple[str, List[Dict]]:
    for node in et:
        entity_start = len(text)
        text += node.text if node.text else ""
        text, entity_list = recursive_text_extract_flattened(
            node,
            entity_prefixes=entity_prefixes,
            text=text,
            entity_list=entity_list
        )
        if any(entity_prefix in node.attrib.get("name", "") for entity_prefix in entity_prefixes):
            entity_list.append({
                "start": entity_start,
                "end": len(text)
            })
            entity_list[-1].update(node.attrib)
        text += node.tail if node.tail else ""
        if node.tag[-4:] == "span":
            text += "\n"
    return text, entity_list


def main():
    import time

    # https://github.com/areed1192/sigma_coding_youtube/blob/master/python/python-finance/sec-web-scraping/Web%20Scraping%20SEC%20-%20XBRL%20Documents.py

    start = time.time()

    file_htm = "/cluster/edgar_filings/aapl-20200926.htm"
    file_cal = "/cluster/edgar_filings/aapl-20200926_cal.xml"
    file_lab = "/cluster/edgar_filings/aapl-20200926_lab.xml"
    file_def = "/cluster/edgar_filings/aapl-20200926_def.xml"

    # Initalize storage units, one will be the master list, one will store all the values, and one will store all GAAP
    # info.
    storage_list = []
    storage_values = {}
    storage_gaap = {}

    # Create a named tuple
    FilingTuple = collections.namedtuple("FilingTuple", ["file_path", "namespace_root", "namespace_label"])

    # Initalize my list of named tuples, I plan to parse.
    files_list = [
        FilingTuple(file_cal, r"{http://www.xbrl.org/2003/linkbase}calculationLink", "calculation"),
        FilingTuple(file_def, r"{http://www.xbrl.org/2003/linkbase}definitionLink", "definition"),
        FilingTuple(file_lab, r"{http://www.xbrl.org/2003/linkbase}labelLink", "label")
    ]

    # Labels come in two forms, those I want and those I don't want.
    avoids = ["linkbase", "roleRef"]
    parse = ["label", "labelLink", "labelArc", "loc", "definitionLink", "definitionArc", "calculationArc"]

    # loop through each file.
    for file in files_list:

        # Parse the tree by passing through the file.
        tree = ElementTree.parse(file.file_path)

        # Grab all the namespace elements we want.
        elements = tree.findall(file.namespace_root)

        # Loop throught each element that was found.
        for element in elements:

            # if the element has children we need to loop through those.
            for child_element in element.iter():

                # split the label to remove the namespace component, this will return a list.
                element_split_label = child_element.tag.split('}')

                # The first element is the namespace, and the second element is a label.
                namespace = element_split_label[0]
                label = element_split_label[1]

                # if it's a label we want then continue.
                if label in parse:

                    # define the item type label
                    element_type_label = file.namespace_label + '_' + label

                    # initalize a smaller dictionary that will house all the content from that element.
                    dict_storage = {"item_type": element_type_label}

                    # grab the attribute keys
                    cal_keys = child_element.keys()

                    # for each key.
                    for key in cal_keys:

                        # parse if needed.
                        if "}" in key:

                            # add the new key to the dictionary and grab the old value.
                            new_key = key.split("}")[1]
                            dict_storage[new_key] = child_element.attrib[key]

                        else:
                            # grab the value.
                            dict_storage[key] = child_element.attrib[key]

                    # At this stage I need to create my master list of IDs which is very important to program. I only
                    # want unique values.
                    # I'm still experimenting with this one but I find `Label` XML file provides the best results.
                    if element_type_label == "label_label":
                        # Grab the Old Label ID for example,
                        # `lab_us-gaap_AllocatedShareBasedCompensationExpense_E5D37E400FB5193199CFCB477063C5EB`
                        key_store = dict_storage["label"]

                        # Create the Master Key, now it's this:
                        # `us-gaap_AllocatedShareBasedCompensationExpense_E5D37E400FB5193199CFCB477063C5EB`
                        master_key = key_store.replace("lab_", "")

                        # Split the Key, now it's this:
                        # ['us-gaap', 'AllocatedShareBasedCompensationExpense', 'E5D37E400FB5193199CFCB477063C5EB']
                        label_split = master_key.split("_")

                        # Create the GAAP ID, now it's this: 'us-gaap:AllocatedShareBasedCompensationExpense'
                        gaap_id = label_split[0] + ":" + label_split[1]

                        # One Dictionary contains only the values from the XML Files.
                        storage_values[master_key] = {}
                        storage_values[master_key]["label_id"] = key_store
                        storage_values[master_key]["location_id"] = key_store.replace("lab_", "loc_")
                        storage_values[master_key]["us_gaap_id"] = gaap_id
                        storage_values[master_key]["us_gaap_value"] = None
                        storage_values[master_key][element_type_label] = dict_storage

                        # The other dicitonary will only contain the values related to GAAP Metrics.
                        storage_gaap[gaap_id] = {}
                        storage_gaap[gaap_id]["id"] = gaap_id
                        storage_gaap[gaap_id]["master_id"] = master_key

                    # add to dictionary.
                    storage_list.append([file.namespace_label, dict_storage])

    # single xml file
    tree = ElementTree.parse(file_htm)
    root = tree.getroot()
    doc = []
    for node in root.iter():
        tag = re.sub("(?={)(.*)(?<=})", "", node.tag)
        attributes = node.attrib
        if tag == "nonNumeric":
            print(tag, attributes)
            name = attributes.get("name")
            if "textblock" in name.lower():
                # text = child.text if child.text else ""
                # text, entity_list = recursive_text_extract(
                #     child,
                #     entity_prefixes=ENTITY_PREFIXES,
                #     entity_list=[],
                #     text=text
                # )
                # entity_list.append({
                #     "start": 0,
                #     "end": len(text),
                # })
                # entity_list[-1].update(attributes)
                text = {
                    "text": node.text if node.text else "",
                    "sub": recursive_text_extract(
                        node,
                        entity_prefixes=ENTITY_PREFIXES + ["aapl"],
                        entity_formats=ENTITY_FORMATS
                    )
                }

                if any(entity_prefix in node.attrib.get("name", "") for entity_prefix in ENTITY_PREFIXES):
                    text["entity"] = node.attrib
                else:
                    text["entity"] = None
                text["tail"] = node.tail if node.tail else ""
                doc.append(text)
                # text = ""
                # for child2 in child1.iter():
                #     if child2.text:
                #         text += child2.text

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
