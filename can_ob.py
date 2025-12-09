import os
from robosuite.models.objects import MujocoXMLObject
import xml.etree.ElementTree as ET


class CanObject(MujocoXMLObject):
    def __init__(self, name="can"):
        xml_path = "can_18/model.xml"
        super().__init__(
            fname=xml_path,
            name=name,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


