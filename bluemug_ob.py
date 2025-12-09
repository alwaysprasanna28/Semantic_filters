import os
from robosuite.models.objects import MujocoXMLObject
import xml.etree.ElementTree as ET


class BlueMugObject(MujocoXMLObject):
    def __init__(self, name="bluemug"):
        xml_path = "Cole_Hardware_Mug_Classic_Blue/bluemug.xml"
        # xml_path = '/home/prasanna/llc/sim/more_fresh/Lenovo_Yoga_2_11/lenovo_yoga.xml'
        super().__init__(
            fname=xml_path,
            name=name,
            obj_type="all",
            duplicate_collision_geoms=True,
        )



bluemug = BlueMugObject()
# print(bluemug.get_xml())