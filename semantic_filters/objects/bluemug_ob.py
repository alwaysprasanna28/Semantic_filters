import os
from robosuite.models.objects import MujocoXMLObject
import xml.etree.ElementTree as ET


class BlueMugObject(MujocoXMLObject):
    def __init__(self, name="bluemug"):
        xml_path = "semantic_filters//objects//Cole_Hardware_Mug_Classic_Blue//bluemug.xml"
        super().__init__(
            fname=xml_path,
            name=name,
            obj_type="all",
            duplicate_collision_geoms=True,
        )



bluemug = BlueMugObject()
print(bluemug.get_xml())