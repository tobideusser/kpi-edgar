import logging
import re
import xml.etree.ElementTree as ElementTree
from typing import Tuple, List, Dict


logger = logging.getLogger(__name__)

ENTITY_PREFIXES = ["us-gaap"]


def recursive_text_extract(
        et: ElementTree,
        entity_prefixes: List[str]
) -> List:
    text_parts = []
    for node in et:
        text_part = {
            "text": node.text if node.text else "",
            "sub": recursive_text_extract(node, entity_prefixes=entity_prefixes)
        }

        if any(entity_prefix in node.attrib.get("name", "") for entity_prefix in entity_prefixes):
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

    start = time.time()

    # single xml file
    tree = ElementTree.parse("/cluster/edgar_filings/aapl-20200926.htm")
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
                    "sub": recursive_text_extract(node, entity_prefixes=ENTITY_PREFIXES + ["aapl"])
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

    # corpus = from_edgar(
    #     raw_data_path="/cluster/edgar_filings/aapl-20200926_htm.xml"
    # )
    # corpus = from_banz_full(
    #     banz_raw_data_path="/shared_lt/ali/kpi_relation_extractor/banz/raw_data",
    #     banz_section_annotation_path="/shared_lt/ali/kpi_relation_extractor/banz/raw_data/pwc_annotations/banz_anhang_section_annotations_14102019_v3.xlsx"
    # )
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
