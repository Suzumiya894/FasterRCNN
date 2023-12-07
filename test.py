import os
from glob import glob
import xml.etree.ElementTree as ET

filepaths = glob(os.path.join("VOCdevkit", "VOC2007", "Annotations", "*"))
for annotation_file in filepaths:

    tree = ET.parse(annotation_file)
    root = tree.getroot()
    assert tree != None, "Failed to parse %s" % annotation_file
    assert len(root.findall("size")) == 1
    size = root.find("size")
    assert len(size.findall("depth")) == 1
    depth = int(size.find("depth").text)
    assert depth == 3
    boxes = []

    import pdb; pdb.set_trace()
    if len(root.findall("object")) == 0 :
        print(annotation_file)