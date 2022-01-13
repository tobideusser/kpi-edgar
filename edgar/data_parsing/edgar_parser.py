import collections
import html
import logging
import os
import re
import unicodedata
import xml.etree.ElementTree as ElementTree
from typing import List, Dict, Optional, Tuple, Union

from lxml import html as lh
from tqdm import tqdm

from edgar.data_classes.data_classes import EdgarEntity, Paragraph, Document, Corpus, Table

logger = logging.getLogger(__name__)


class EdgarParser:
    def __init__(
            self,
            entity_prefixes: List[str],
            entity_formats: List[str],
            path_to_data_folders: str,
            debug_size: Optional[int] = None,
            dataset_name: Optional[str] = None
    ):
        self.entity_prefixes = entity_prefixes
        self.entity_formats = entity_formats
        self.path_to_data_folders = path_to_data_folders
        self.debug_size = debug_size
        self.dataset_name = dataset_name if dataset_name else "EDGAR"

    def _recursive_text_extract(
            self,
            et: ElementTree,
            storage_gaap: Dict,
            storage_values: Dict
    ) -> List:
        text_parts = []
        for node in et:
            text_part = {
                "text": html.unescape(node.text).strip().replace("\xa0", " ") if node.text else "",
                "sub": self._recursive_text_extract(
                    et=node,
                    storage_gaap=storage_gaap,
                    storage_values=storage_values
                )
            }

            if any(entity_prefix == node.attrib.get("name", "") for entity_prefix in self.entity_prefixes) or \
                    any(entity_format == node.attrib.get("format", "") for entity_format in self.entity_formats):
                text_part["entity"] = node.attrib
                text_part["entity"]["gaap"] = storage_gaap.get(text_part["entity"]["name"], None)
                if text_part["entity"]["gaap"] is not None:
                    text_part["entity"]["value"] = storage_values.get(text_part["entity"]["gaap"]["master_id"], None)
            else:
                text_part["entity"] = None
            text_part["tail"] = html.unescape(node.tail).strip().replace("\xa0", " ") if node.tail else ""
            if text_part["text"] != "" or text_part["tail"] != "" or text_part["entity"] is not None:
                text_parts.append(text_part)
            elif len(text_part["sub"]) > 0:
                text_parts.append(text_part["sub"])
        return text_parts

    def _parse_edgar_entry(
            self,
            file_htm: str,
            file_cal: str,
            file_lab: str,
            file_def: str
    ) -> Dict[str, List]:
        # file_htm = "/cluster/edgar_filings/aapl-20200926.htm"
        # file_cal = "/cluster/edgar_filings/aapl-20200926_cal.xml"
        # file_lab = "/cluster/edgar_filings/aapl-20200926_lab.xml"
        # file_def = "/cluster/edgar_filings/aapl-20200926_def.xml"

        # Initalize storage units, one will be the master list, one will store all the values, and one will store all
        # GAAP info.
        storage_list = []
        storage_values = {}
        storage_gaap = {}

        # Create a named tuple
        FilingTuple = collections.namedtuple("FilingTuple", ["raw_string", "namespace_root", "namespace_label"])

        # Initalize my list of named tuples, I plan to parse.
        files_list = [
            FilingTuple(file_cal, r"{http://www.xbrl.org/2003/linkbase}calculationLink", "calculation"),
            FilingTuple(file_def, r"{http://www.xbrl.org/2003/linkbase}definitionLink", "definition"),
            FilingTuple(file_lab, r"{http://www.xbrl.org/2003/linkbase}labelLink", "label")
        ]

        # Labels come in two forms, those I want and those I don't want.
        # avoids = ["linkbase", "roleRef"]
        parse = ["label", "labelLink", "labelArc", "loc", "definitionLink", "definitionArc", "calculationArc"]

        # loop through each file.
        for file in files_list:

            # Parse the tree by passing through the file.
            tree = ElementTree.ElementTree(ElementTree.fromstring(file.raw_string))
            # tree = ElementTree.parse(file.file_path)

            # Grab all the namespace elements we want.
            elements = tree.findall(file.namespace_root)

            # Loop throught each element that was found.
            for element in elements:

                # if the element has children we need to loop through those.
                for child_element in element.iter():

                    # split the label to remove the namespace component, this will return a list.
                    element_split_label = child_element.tag.split('}')

                    # The first element is the namespace, and the second element is a label.
                    # namespace = element_split_label[0]
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

                        # At this stage I need to create my master list of IDs which is very important to program. I
                        # only want unique values.
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

        doc = {"text": [], "table": []}

        # parsing tables first
        tree = lh.fromstring(bytes(file_htm, encoding="utf8"))
        tables = tree.xpath('//table')
        for tab in tables:
            table = Table.from_html(tab)
            doc["table"].append(table)

        tree = ElementTree.ElementTree(ElementTree.fromstring(file_htm))
        root = tree.getroot()

        for node in root.iter():
            tag = re.sub("(?={)(.*)(?<=})", "", node.tag)
            attributes = node.attrib
            if tag == "nonNumeric":
                # print(tag, attributes)
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
                        "text": html.unescape(node.text).strip().replace("\xa0", " ") if node.text else "",
                        "sub": self._recursive_text_extract(
                            node,
                            storage_gaap=storage_gaap,
                            storage_values=storage_values
                        )
                    }

                    if any(entity_prefix in node.attrib.get("name", "") for entity_prefix in self.entity_prefixes) or \
                            any(entity_ft == node.attrib.get("format", "") for entity_ft in self.entity_formats):
                        text["entity"] = node.attrib
                        text["entity"]["gaap"] = storage_gaap.get(text["entity"]["name"], None)
                        if text["entity"]["gaap"] is not None:
                            text["entity"]["value"] = storage_values.get(text["entity"]["gaap"]["master_id"], None)
                    else:
                        text["entity"] = None
                    text["tail"] = html.unescape(node.tail).strip().replace("\xa0", " ") if node.tail else ""
                    doc["text"].append(text)
        return doc

    @staticmethod
    def recursive_transform_subparts(
            sub: List[Union[Dict, List]],
            text: str = "",
            entities: Optional[List[EdgarEntity]] = None
    ) -> Tuple[str, List[EdgarEntity]]:

        if entities is None:
            entities = []

        for text_part in sub:

            if isinstance(text_part, dict):
                # if dict, extract the information stored therein

                start = len(text)
                text += text_part["text"] + (" " if text_part["text"] else "")

                if text_part["entity"]:
                    entities.append(EdgarEntity(
                        start=start,
                        end=len(text),
                        id_=text_part["entity"]["id"],
                        name=text_part["entity"]["name"],
                        value=text_part["text"],
                        context_ref=text_part["entity"].get("contextRef", None),
                        continued_at=text_part["entity"].get("continuedAt", None),
                        escape=bool(text_part["entity"].get("continuedAt", None)),
                        gaap_id=text_part["entity"]["gaap"].get("id", None),
                        gaap_master_id=text_part["entity"]["gaap"].get("master_id", None),
                        label_id=text_part["entity"]["value"].get("label_id", None),
                        location_id=text_part["entity"]["value"].get("location_id", None),
                        us_gaap_id=text_part["entity"]["value"].get("us_gaap_id", None),
                        us_gaap_value=text_part["entity"]["value"].get("us_gaap_value", None)
                    ))

                if text_part["sub"]:
                    text, entities = EdgarParser.recursive_transform_subparts(
                        sub=text_part["sub"],
                        text=text,
                        entities=entities
                    )
                text += text_part["tail"] + (" " if text_part["tail"] else "")

            else:
                # assume text_part is a list, pass the whole list to recursive_transform_subparts() to loop through
                # it again
                text, entities = EdgarParser.recursive_transform_subparts(
                    sub=text_part,
                    text=text,
                    entities=entities
                )
        return text, entities

    def transform_to_dataclasses(self, parsed_data: Dict) -> Corpus:

        documents = []
        for key, doc in parsed_data.items():

            i = 0
            segments = []
            for seg in doc["text"]:

                # compress all subparts into one string
                text, entities = EdgarParser.recursive_transform_subparts(seg["sub"])
                text = text[:-1] if len(text) > 0 and text[-1] == " " else text

                # check for a EdgarEntity, i.e. a entity / type of the text block
                if seg["entity"]:
                    textblock_entity = EdgarEntity(
                        value=text,
                        name=seg["entity"]["name"],
                        id_=seg["entity"]["id"],
                        start=0,
                        end=len(text),
                        context_ref=seg["entity"].get("contextRef", None),
                        continued_at=seg["entity"].get("continuedAt", None),
                        escape=bool(seg["entity"].get("continuedAt", None)),
                        gaap_id=seg["entity"]["gaap"].get("id", None),
                        gaap_master_id=seg["entity"]["gaap"].get("master_id", None),
                        label_id=seg["entity"]["value"].get("label_id", None),
                        location_id=seg["entity"]["value"].get("location_id", None),
                        us_gaap_id=seg["entity"]["value"].get("us_gaap_id", None),
                        us_gaap_value=seg["entity"]["value"].get("us_gaap_value", None)
                    )
                else:
                    textblock_entity = None
                segments.append(Paragraph(
                    id_=i,
                    value=text,
                    textblock_entity=textblock_entity,
                    edgar_entities=entities
                ))
                i += 1
            documents.append(Document(
                id_=key,
                segments=segments
            ))

        corpus = Corpus(documents=documents, name=self.dataset_name)

        return corpus

    def parse_data_folder(self):
        parsed_data = dict()
        i = 0
        pbar = tqdm(desc="Parsing txt files")
        for subdir, dirs, files in os.walk(self.path_to_data_folders):
            if self.debug_size and self.debug_size <= i:
                break
            for file in files:
                if self.debug_size and self.debug_size <= i:
                    break
                if os.path.splitext(file)[1] == ".txt":
                    with open(os.path.join(subdir, file)) as f:
                        raw_content = f.read()
                    raw_content = re.sub("<DOCUMENT>\n<TYPE>GRAPHIC.*?</DOCUMENT>", "", raw_content, flags=re.DOTALL)
                    raw_content = re.sub("<DOCUMENT>\n<TYPE>ZIP.*?</DOCUMENT>", "", raw_content, flags=re.DOTALL)
                    raw_content = re.sub("<DOCUMENT>\n<TYPE>EXCEL.*?</DOCUMENT>", "", raw_content, flags=re.DOTALL)
                    raw_content = re.sub("<DOCUMENT>\n<TYPE>XML.*?</DOCUMENT>", "", raw_content, flags=re.DOTALL)

                    raw_content = unicodedata.normalize("NFKC", raw_content)

                    # this is the regex solution, probably more prone to errors, but lxml seems to be bugged when
                    # converting back to string

                    # get all documents
                    all_docs = re.findall(
                        r"(?i)(?<=<DOCUMENT>)(.*?)(?=</DOCUMENT>)",
                        string=raw_content,
                        flags=re.DOTALL
                    )

                    # dict to store the content split into relevant sections
                    split_content = dict()

                    # loop through all documents
                    for doc in all_docs:

                        # get the type
                        doc_type = re.search(
                            r"(?i)(?<=<TYPE>)(.*?)(?=<)",
                            string=doc,
                            flags=re.DOTALL
                        )[0]

                        if "10-K" in doc_type.upper():

                            split_content["10-K"] = re.search(
                                r"(?i)(?<=<XBRL>)(.*?)(?=</XBRL>)",
                                string=doc,
                                flags=re.DOTALL
                            )[0]
                            # remove leading linebreaks if they exist
                            while split_content["10-K"][0] == "\n":
                                split_content["10-K"] = split_content["10-K"][1:]

                        elif "CAL" in doc_type.upper():

                            split_content["CAL"] = re.search(
                                r"(?i)(?<=<XBRL>)(.*?)(?=</XBRL>)",
                                string=doc,
                                flags=re.DOTALL
                            )[0]
                            # remove leading linebreaks if they exist
                            while split_content["CAL"][0] == "\n":
                                split_content["CAL"] = split_content["CAL"][1:]

                        elif "LAB" in doc_type.upper():

                            split_content["LAB"] = re.search(
                                r"(?i)(?<=<XBRL>)(.*?)(?=</XBRL>)",
                                string=doc,
                                flags=re.DOTALL
                            )[0]
                            # remove leading linebreaks if they exist
                            while split_content["LAB"][0] == "\n":
                                split_content["LAB"] = split_content["LAB"][1:]

                        elif "DEF" in doc_type.upper():

                            split_content["DEF"] = re.search(
                                r"(?i)(?<=<XBRL>)(.*?)(?=</XBRL>)",
                                string=doc,
                                flags=re.DOTALL
                            )[0]
                            # remove leading linebreaks if they exist
                            while split_content["DEF"][0] == "\n":
                                split_content["DEF"] = split_content["DEF"][1:]

                    # if condition: only parse whole edgar entry if the four relevant reports were found
                    if all(a in list(split_content.keys()) for a in ["10-K", "CAL", "DEF", "LAB"]):
                        unique_dict_key = "_".join(os.path.normpath(subdir).split(os.path.sep)[-2:]) + "_" + file
                        parsed_data[unique_dict_key] = self._parse_edgar_entry(
                            file_htm=split_content["10-K"],
                            file_cal=split_content["CAL"],
                            file_lab=split_content["LAB"],
                            file_def=split_content["DEF"]
                        )
                        i += 1
                        pbar.update(1)

        transformed_data = self.transform_to_dataclasses(parsed_data=parsed_data)

        return transformed_data


def main():
    ep = EdgarParser(
        entity_prefixes=["us-gaap"],
        entity_formats=["ixt:numdotdecimal", "ix:nonFraction"],
        debug_size=10,
        path_to_data_folders="/cluster/debug_edgar_filings"
    )
    ep.parse_data_folder()


if __name__ == "__main__":
    main()
