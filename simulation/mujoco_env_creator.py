import os
import yaml
import mujoco
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple
import glog as log
import mujoco.viewer
import argparse


class MujocoEnvCreator:
    """
    A class for creating MuJoCo environments with configurable robots and scene objects.
    Uses template-based configuration system for all settings.
    """
    
    def __init__(self, config_path: str = "simulation/scene_config/example.yaml", 
                 template_path: str = "assets/scene_templates/scene_template.xml"):
        """Initialize the MuJoCo environment creator."""
        self.config_path = config_path
        self.template_path = template_path
        self.root_path = Path(__file__).parent.parent
        self.config = self._load_config()
        self.template_config = self._load_template_config()
        self.mj_model = None
        self.mj_data = None
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(self.config_path)
        if not config_path.is_absolute():
            config_path = self.root_path / config_path
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_template_config(self) -> ET.Element:
        """Load XML template configuration."""
        template_path = Path(self.template_path)
        if not template_path.is_absolute():
            template_path = self.root_path / template_path
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        return ET.parse(template_path).getroot()
    
    def _get_full_path(self, relative_path: str) -> str:
        """Convert relative path to full path."""
        return str(self.root_path / relative_path)
    
    def _get_full_path_if_exists(self, relative_path: str) -> str:
        """Get full path and check if file exists."""
        full_path = self._get_full_path(relative_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        return full_path
    
    def _copy_element_tree(self, source: ET.Element, target: ET.Element) -> None:
        """Recursively copy XML elements from source to target."""
        for child in source:
            new_child = ET.SubElement(target, child.tag)
            new_child.attrib.update(child.attrib)
            if child.text and child.text.strip():
                new_child.text = child.text
            self._copy_element_tree(child, new_child)
    
    def _copy_element_tree_without_free_joints(self, source: ET.Element, target: ET.Element) -> None:
        """Recursively copy XML elements from source to target, skipping free joints."""
        for child in source:
            # Skip free joints to avoid conflicts
            if child.tag == "joint" and child.get("type") == "free":
                continue
            
            new_child = ET.SubElement(target, child.tag)
            new_child.attrib.update(child.attrib)
            if child.text and child.text.strip():
                new_child.text = child.text
            self._copy_element_tree_without_free_joints(child, new_child)
    
    def _robot_has_free_joint(self, robot_xml: ET.Element) -> bool:
        """Check if robot XML already contains a free joint."""
        for elem in robot_xml.iter():
            if elem.tag == "joint" and elem.get("type") == "free":
                return True
        return False
    
    def _copy_worldbody_with_position_offset(self, source_worldbody: ET.Element, target_worldbody: ET.Element, 
                                           pos_offset: list, quat_offset: list) -> None:
        """Copy worldbody content with position offset for robots that already have free joints."""
        for child in source_worldbody:
            new_child = ET.SubElement(target_worldbody, child.tag)
            new_child.attrib.update(child.attrib)
            
            # Apply position offset to the first body (usually the base)
            if child.tag == "body":
                current_pos = child.get("pos", "0 0 0").split()
                try:
                    new_pos = [float(current_pos[i]) + pos_offset[i] for i in range(3)]
                    new_child.set("pos", " ".join(map(str, new_pos)))
                except (IndexError, ValueError):
                    new_child.set("pos", " ".join(map(str, pos_offset)))
                    
                # Apply quaternion if needed (for now, just use the offset quat)
                if quat_offset != [0.0, 0.0, 0.0, 1.0]:
                    new_child.set("quat", " ".join(map(str, quat_offset)))
            
            if child.text and child.text.strip():
                new_child.text = child.text
            self._copy_element_tree(child, new_child)
    
    def _asset_exists(self, asset_parent: ET.Element, new_asset: ET.Element) -> bool:
        """Check if an asset with the same type and name already exists."""
        new_asset_name = new_asset.get('name')
        new_asset_tag = new_asset.tag
        
        if not new_asset_name:
            return False
        
        for existing_asset in asset_parent:
            if (existing_asset.tag == new_asset_tag and 
                existing_asset.get('name') == new_asset_name):
                return True
        
        return False
    
    def _add_template_section(self, root: ET.Element, section_name: str) -> None:
        """Add a section from template to root."""
        template_section = self.template_config.find(section_name)
        if template_section is not None:
            new_section = ET.SubElement(root, section_name)
            
            # Handle special cases for paths
            if section_name == "compiler":
                for attr_name, attr_value in template_section.attrib.items():
                    if attr_name in ['meshdir', 'texturedir']:
                        new_section.set(attr_name, str(self.root_path / attr_value))
                    else:
                        new_section.set(attr_name, attr_value)
            else:
                # Copy all attributes and children
                new_section.attrib.update(template_section.attrib)
                self._copy_element_tree(template_section, new_section)
    
    def _add_robot_assets_and_worldbody(self, root: ET.Element, worldbody: ET.Element) -> None:
        """Add robot assets and worldbody content using proper XML merging and attach mechanism."""
        robot_config = self.config.get("robot", {})
        robot_path = robot_config.get("model_path", "")
        
        if not robot_path:
            log.warning("No robot model path specified in configuration")
            return
        
        try:
            full_robot_path = self._get_full_path_if_exists(robot_path)
        except FileNotFoundError as e:
            log.error(str(e))
            return
        
        # Parse robot XML and merge into main XML
        robot_xml = ET.parse(full_robot_path).getroot()
        
        # Merge robot assets and defaults into main XML
        self._merge_assets(root, robot_xml)
        self._merge_defaults(root, robot_xml)
        
        # Handle robot positioning and base configuration
        fix_base = robot_config.get("fix_base", True)
        initial_pos = robot_config.get("initial_position", [0.0, 0.0, 0.0])
        initial_quat = robot_config.get("initial_orientation", [0.0, 0.0, 0.0, 1.0])
        robot_name = robot_config.get("name", "robot")
        
        # Check if robot already has a free joint
        has_free_joint = self._robot_has_free_joint(robot_xml)
        
        # Only create container body if robot doesn't already have free joint or if it's fixed base
        if fix_base or not has_free_joint:
            robot_body = ET.SubElement(worldbody, "body")
            robot_body.set("name", f"{robot_name}_base")
            robot_body.set("pos", " ".join(map(str, initial_pos)))
            robot_body.set("quat", " ".join(map(str, initial_quat)))
            
            if not fix_base and not has_free_joint:
                # Add minimal inertial properties for free-floating robots
                inertial = ET.SubElement(robot_body, "inertial")
                inertial.set("pos", "0 0 0")
                inertial.set("mass", "0.001")  # Minimal mass
                inertial.set("diaginertia", "1e-6 1e-6 1e-6")  # Minimal inertia
                
                free_joint = ET.SubElement(robot_body, "joint")
                free_joint.set("name", f"{robot_name}_free_joint")
                free_joint.set("type", "free")
        else:
            robot_body = None  # No container needed
        
        # Copy robot worldbody content, handling free joints properly
        robot_worldbody = robot_xml.find("worldbody")
        if robot_worldbody is not None:
            if not fix_base and has_free_joint:
                # For robots that already have free joints (like G1), copy directly to worldbody with position offset
                self._copy_worldbody_with_position_offset(robot_worldbody, worldbody, initial_pos, initial_quat)
            elif robot_body is not None:
                if not fix_base:
                    # For free-floating robots without existing free joint, copy content without adding another free joint
                    self._copy_element_tree_without_free_joints(robot_worldbody, robot_body)
                else:
                    # For fixed robots, copy normally
                    self._copy_element_tree(robot_worldbody, robot_body)
        
        # Merge robot actuators
        self._merge_actuators(root, robot_xml)
        
        # Add gripper attachments using attach mechanism
        self._add_gripper_attachments(root, robot_config)
        
        log.info(f"Added robot {robot_config.get('name', 'robot')} with XML merging from {robot_path}")
    
    
    
    def _add_scene_objects_to_worldbody(self, worldbody: ET.Element) -> None:
        """Add scene objects to worldbody using unified rigid body approach."""
        scene_objects = self.config.get("scene_objects", [])
        
        for obj_config in scene_objects:
            obj_name = obj_config.get("name", "object")
            obj_type = obj_config.get("type", "unknown")
            
            try:
                self._add_rigid_body_object(worldbody, obj_config)
                fix_base = obj_config.get("fix_base", True)
                base_type = "fixed" if fix_base else "floating"
                log.info(f"Added {obj_type} object: {obj_name} with {base_type} base")
            except Exception as e:
                log.error(f"Failed to add object {obj_name}: {e}")
    
    def _add_rigid_body_object(self, worldbody: ET.Element, obj_config: Dict) -> None:
        """Add a rigid body object (table, box, etc.) to the worldbody."""
        obj_name = obj_config.get("name", "object")
        initial_pos = obj_config.get("initial_position", [0.0, 0.0, 0.0])
        initial_quat = obj_config.get("initial_orientation", [0.0, 0.0, 0.0, 1.0])
        fix_base = obj_config.get("fix_base", True)
        
        # Create body
        body = ET.SubElement(worldbody, "body")
        body.set("name", obj_name)
        body.set("pos", " ".join(map(str, initial_pos)))
        body.set("quat", " ".join(map(str, initial_quat)))
        
        # Add free joint for floating objects
        if not fix_base:
            joint = ET.SubElement(body, "joint")
            joint.set("name", f"{obj_name}_joint")
            joint.set("type", "free")
            joint.set("damping", "0.001")
        
        # Add geometry based on type
        self._add_geometry(body, obj_config)
    
    def _add_geometry(self, body: ET.Element, obj_config: Dict) -> None:
        """Add geometry to body based on object configuration."""
        obj_name = obj_config.get("name", "object")
        model_path = obj_config.get("model_path", "")
        
        if "" == model_path:
            log.error(f"Model path not specified for object {obj_name}")

        # Use URDF/XML model
        full_path = self._get_full_path(model_path)
        if os.path.exists(full_path):
            if full_path.endswith('.xml'):
                self._inline_xml_content(body, full_path, obj_name)
            else:
                # For non-XML files, create an include with a prefixed doclass
                include = ET.SubElement(body, "include")
                include.set("file", model_path)
                # Prefix doclass to avoid conflicts with main scene defaults
                if obj_config.get("doclass", ""):
                    include.set("doclass", f"{obj_name}_{obj_config['doclass']}")
        else:
            raise FileNotFoundError(f"Model file not found: {full_path}")
    
    def _inline_xml_content(self, parent_body: ET.Element, xml_path: str, obj_name: str) -> None:
        """Inline XML content directly into the parent body."""
        try:
            with open(xml_path, 'r') as f:
                xml_content = f.read()
            
            temp_xml = f"<temp>{xml_content}</temp>"
            temp_root = ET.fromstring(temp_xml)
            
            for child in temp_root:
                if child.tag == "mujoco":
                    for element in child:
                        if element.tag in ["inertial", "geom", "joint", "body"]:
                            new_element = ET.SubElement(parent_body, element.tag)
                            for attr_name, attr_value in element.attrib.items():
                                if attr_name == "name":
                                    new_element.set(attr_name, f"{obj_name}_{attr_value}")
                                else:
                                    new_element.set(attr_name, attr_value)
                            if element.text and element.text.strip():
                                new_element.text = element.text
                        
                        elif element.tag == "worldbody":
                            for body_element in element:
                                if body_element.tag == "body":
                                    for sub_element in body_element:
                                        new_element = ET.SubElement(parent_body, sub_element.tag)
                                        for attr_name, attr_value in sub_element.attrib.items():
                                            if attr_name == "name":
                                                new_element.set(attr_name, f"{obj_name}_{attr_value}")
                                            else:
                                                new_element.set(attr_name, attr_value)
                                        if sub_element.text and sub_element.text.strip():
                                            new_element.text = sub_element.text
                    break
                    
        except Exception as e:
            log.error(f"Error inlining XML content: {e}")
            raise RuntimeError(f"Failed to parse XML content for object {obj_name}: {e}")
    
    def _add_keyframes_from_config(self, root: ET.Element) -> None:
        """Add keyframes from YAML configuration to the XML (legacy method)."""
        keyframes_config = self.config.get("keyframes", [])
        
        if not keyframes_config:
            return
        
        # Find or create keyframe section
        keyframe_element = root.find("keyframe")
        if keyframe_element is None:
            keyframe_element = ET.SubElement(root, "keyframe")
        
        # Add each keyframe from configuration
        for keyframe_config in keyframes_config:
            key_element = ET.SubElement(keyframe_element, "key")
            
            # Set keyframe attributes
            key_element.set("name", keyframe_config.get("name", "unnamed"))
            key_element.set("qpos", keyframe_config.get("qpos", ""))
            key_element.set("time", str(keyframe_config.get("time", "0")))
            key_element.set("ctrl", keyframe_config.get("ctrl", ""))
            key_element.set("qvel", keyframe_config.get("qvel", ""))
            
            log.info(f"Added keyframe: {keyframe_config.get('name', 'unnamed')}")
    
    def _merge_keyframes_dynamically(self, root: ET.Element) -> None:
        """Dynamically merge keyframes from robot and gripper XML files."""
        # Find or create keyframe section
        keyframe_element = root.find("keyframe")
        if keyframe_element is None:
            keyframe_element = ET.SubElement(root, "keyframe")
        
        # Merge robot keyframes
        self._merge_robot_keyframes(keyframe_element)
        
        # Merge gripper keyframes
        self._merge_gripper_keyframes(keyframe_element)
        
        log.info("Dynamically merged keyframes from robot and grippers")
    
    def _merge_robot_keyframes(self, keyframe_element: ET.Element) -> None:
        """Merge keyframes from robot XML file."""
        robot_config = self.config.get("robot", {})
        robot_path = robot_config.get("model_path", "")
        
        if not robot_path:
            return
        
        full_robot_path = self._get_full_path(robot_path)
        if not os.path.exists(full_robot_path):
            return
        
        try:
            robot_xml = ET.parse(full_robot_path).getroot()
            robot_keyframes = robot_xml.find("keyframe")
            
            if robot_keyframes is not None:
                for key in robot_keyframes.findall("key"):
                    # Create new key element
                    new_key = ET.SubElement(keyframe_element, "key")
                    # Copy all attributes
                    for attr_name, attr_value in key.attrib.items():
                        new_key.set(attr_name, attr_value)
                    
                    log.info(f"Merged robot keyframe: {key.get('name', 'unnamed')}")
        except Exception as e:
            log.error(f"Failed to merge robot keyframes: {e}")
    
    def _merge_gripper_keyframes(self, keyframe_element: ET.Element) -> None:
        """Merge keyframes from gripper XML files with proper joint prefixing."""
        robot_config = self.config.get("robot", {})
        grippers = robot_config.get("grippers", {})
        
        if not grippers:
            return
        
        for gripper_name, gripper_config in grippers.items():
            gripper_path = gripper_config.get("model_path", "")
            
            if not gripper_path:
                continue
            
            full_gripper_path = self._get_full_path(gripper_path)
            if not os.path.exists(full_gripper_path):
                continue
            
            try:
                gripper_xml = ET.parse(full_gripper_path).getroot()
                gripper_keyframes = gripper_xml.find("keyframe")
                
                if gripper_keyframes is not None:
                    for key in gripper_keyframes.findall("key"):
                        # Create new key element with prefixed name
                        new_key = ET.SubElement(keyframe_element, "key")
                        
                        # Copy and modify attributes
                        for attr_name, attr_value in key.attrib.items():
                            if attr_name == "name":
                                # Use gripper_name instead of name_prefix for uniqueness
                                new_key.set(attr_name, f"{gripper_name}_{attr_value}")
                            else:
                                new_key.set(attr_name, attr_value)
                        
                        log.info(f"Merged gripper keyframe: {gripper_name}_{key.get('name', 'unnamed')}")
            except Exception as e:
                log.error(f"Failed to merge gripper keyframes from {gripper_name}: {e}")
    
    def generate_xml(self) -> str:
        """Generate the complete XML string for the MuJoCo scene."""
        root = ET.Element("mujoco")
        root.set("model", "generated_scene")
        
        # Add all sections from template
        for section in ["compiler", "statistic", "visual", "default", "option", "asset", "worldbody"]:
            self._add_template_section(root, section)
        
        # Get worldbody for robot and objects
        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")
        
        # Add robot and scene objects using include mechanism
        self._add_robot_assets_and_worldbody(root, worldbody)
        self._add_scene_objects_to_worldbody(worldbody)
        
        # Note: No longer adding keyframes to XML, will apply them dynamically to mj_data instead
        
        return ET.tostring(root, encoding='unicode')
    
    def _merge_assets(self, root: ET.Element, source_xml: ET.Element) -> None:
        """Merge asset definitions from source XML into root."""
        main_asset = root.find("asset")
        if main_asset is None:
            main_asset = ET.SubElement(root, "asset")
        
        source_asset = source_xml.find("asset")
        if source_asset is not None:
            for child in source_asset:
                # Check if asset already exists to avoid duplicates
                if not self._asset_exists(main_asset, child):
                    new_asset = ET.SubElement(main_asset, child.tag)
                    new_asset.attrib.update(child.attrib)
                    if child.text and child.text.strip():
                        new_asset.text = child.text
    
    def _merge_defaults(self, root: ET.Element, source_xml: ET.Element) -> None:
        """Merge default class definitions from source XML into root."""
        main_default = root.find("default")
        if main_default is None:
            main_default = ET.SubElement(root, "default")
        
        source_default = source_xml.find("default")
        if source_default is not None:
            for child in source_default:
                if child.tag == "default" and child.get("class"):
                    # Check if class already exists
                    class_name = child.get("class")
                    existing_class = None
                    for existing in main_default.findall("default"):
                        if existing.get("class") == class_name:
                            existing_class = existing
                            break
                    
                    if not existing_class:
                        # Add new default class
                        new_default = ET.SubElement(main_default, "default")
                        new_default.attrib.update(child.attrib)
                        if child.text and child.text.strip():
                            new_default.text = child.text
                        # Copy all children recursively
                        self._copy_element_tree(child, new_default)
    
    def _merge_actuators(self, root: ET.Element, source_xml: ET.Element) -> None:
        """Merge actuator definitions from source XML into root."""
        main_actuator = root.find("actuator")
        if main_actuator is None:
            main_actuator = ET.SubElement(root, "actuator")
        
        source_actuator = source_xml.find("actuator")
        if source_actuator is not None:
            for child in source_actuator:
                new_actuator = ET.SubElement(main_actuator, child.tag)
                new_actuator.attrib.update(child.attrib)
                if child.text and child.text.strip():
                    new_actuator.text = child.text
    
    def _add_gripper_attachments(self, root: ET.Element, robot_config: dict) -> None:
        """Add gripper attachments using MuJoCo's <attach> mechanism."""
        grippers = robot_config.get("grippers", {})
        
        if not grippers:
            log.info("No grippers configured for attachment")
            return
        
        # First, add gripper models as assets
        self._add_gripper_assets(root, grippers)
        
        # Find the worldbody to add attach elements
        worldbody = root.find("worldbody")
        if worldbody is None:
            log.error("No worldbody found for gripper attachment")
            return
        
        # Add attach elements for each gripper
        for gripper_name, gripper_config in grippers.items():
            gripper_path = gripper_config.get("model_path", "")
            attachment_site = gripper_config.get("attachment_site", "")
            gripper_type = gripper_config.get("type", "")
            # Extract position from gripper_name (left_gripper -> left, right_gripper -> right)
            position = gripper_name.split("_")[0] if "_" in gripper_name else gripper_name
            name_prefix = f"{position}_{gripper_type}_" if gripper_type else f"{position}_"
            
            if not gripper_path or not attachment_site:
                log.warning(f"Incomplete gripper configuration for {gripper_name}")
                continue
            
            full_gripper_path = self._get_full_path(gripper_path)
            if not os.path.exists(full_gripper_path):
                log.error(f"Gripper model file not found: {full_gripper_path}")
                continue
            
            # Find the attachment site body
            attachment_body = self._find_site_parent_body(worldbody, attachment_site)
            if attachment_body is None:
                log.error(f"Attachment site {attachment_site} not found")
                continue
            
            # Parse gripper XML to get the attach body name
            gripper_xml = ET.parse(full_gripper_path).getroot()
            attach_body_name = self._get_gripper_attach_body(gripper_xml)
            
            # Create attach element
            attach = ET.SubElement(attachment_body, "attach")
            attach.set("model", gripper_name)  # Use the asset name
            attach.set("prefix", name_prefix)
            
            # Set attachment body from gripper XML
            if attach_body_name:
                attach.set("body", attach_body_name)
                log.info(f"Using attach body: {attach_body_name} from gripper model")
            else:
                log.warning(f"No attach body found in gripper model {gripper_name}, will attach worldbody contents")
            
            log.info(f"Added attach element for gripper: {gripper_name} at site {attachment_site} with prefix {name_prefix}")
    
    def _find_site_parent_body(self, worldbody: ET.Element, site_name: str) -> ET.Element:
        """Find the body that contains the specified site."""
        for body in worldbody.iter("body"):
            for site in body.findall("site"):
                if site.get("name") == site_name:
                    return body
        return None
    
    def _add_gripper_assets(self, root: ET.Element, grippers: dict) -> None:
        """Add gripper models as assets for use with attach mechanism."""
        main_asset = root.find("asset")
        if main_asset is None:
            main_asset = ET.SubElement(root, "asset")
        
        for gripper_name, gripper_config in grippers.items():
            gripper_path = gripper_config.get("model_path", "")
            
            if not gripper_path:
                continue
            
            full_gripper_path = self._get_full_path(gripper_path)
            if not os.path.exists(full_gripper_path):
                log.error(f"Gripper model file not found: {full_gripper_path}")
                continue
            
            # Add gripper model as an asset
            model_asset = ET.SubElement(main_asset, "model")
            model_asset.set("name", gripper_name)
            model_asset.set("file", gripper_path)
            
            log.info(f"Added gripper model asset: {gripper_name} from {gripper_path}")

    def _get_gripper_attach_body(self, gripper_xml: ET.Element) -> str:
        """Get the first body name from gripper worldbody to use as attach body."""
        worldbody = gripper_xml.find("worldbody")
        if worldbody is not None:
            # Find the first body in worldbody
            first_body = worldbody.find("body")
            if first_body is not None:
                body_name = first_body.get("name")
                if body_name:
                    return body_name
        return None
    
    def _copy_element_with_prefix(self, parent: ET.Element, element: ET.Element, name_prefix: str) -> None:
        """Copy element with name prefixing (kept for potential future use)."""
        new_element = ET.SubElement(parent, element.tag)
        for attr_name, attr_value in element.attrib.items():
            if attr_name in ["name", "joint", "mesh", "joint1", "joint2", "body1", "body2"] and attr_value:
                new_element.set(attr_name, f"{name_prefix}{attr_value}")
            else:
                new_element.set(attr_name, attr_value)
        if element.text and element.text.strip():
            new_element.text = element.text
        
        # Recursively copy children
        for child in element:
            self._copy_element_with_prefix(new_element, child, name_prefix)
    
    def save_xml(self, output_path: str) -> None:
        """Save the generated XML to a file with proper formatting."""
        xml_content = self.generate_xml()
        
        from xml.dom import minidom
        dom = minidom.parseString(xml_content)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
        
        with open(output_path, 'w') as f:
            f.write(pretty_xml)
        
        log.info(f"Generated XML saved to: {output_path}")
    
    def create_model(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Create and return MuJoCo model and data objects."""
        xml_content = self.generate_xml()
        
        try:
            # Create the complete model with gripper attachments
            self.mj_model = mujoco.MjModel.from_xml_string(xml_content)
            
            # Gripper attachments are already handled in XML generation
            
            self.mj_data = mujoco.MjData(self.mj_model)
            
            sim_config = self.config.get("simulation", {})
            self.mj_model.opt.timestep = sim_config.get("timestep", 0.001)
            
            # Apply dynamically merged keyframes
            self._apply_dynamic_keyframes()
            
            mujoco.mj_forward(self.mj_model, self.mj_data)
            
            log.info(f"Created MuJoCo model with {self.mj_model.nq} DoFs, {self.mj_model.nu} actuators, {self.mj_model.nkey} keyframes")
            return self.mj_model, self.mj_data
            
        except Exception as e:
            log.error(f"Failed to create MuJoCo model: {e}")
            debug_path = "debug_generated_scene.xml"
            with open(debug_path, 'w') as f:
                f.write(xml_content)
            log.error(f"Debug XML saved to: {debug_path}")
            raise
    
    def _collect_keyframe_data(self) -> Dict[str, Dict[str, any]]:
        """Collect keyframe data from robot and gripper XML files."""
        keyframes = {}
        
        # Collect robot keyframes
        robot_keyframes = self._collect_robot_keyframes()
        keyframes.update(robot_keyframes)
        
        # Collect gripper keyframes
        gripper_keyframes = self._collect_gripper_keyframes()
        keyframes.update(gripper_keyframes)
        
        log.info(f"Collected {len(keyframes)} keyframes from robot and grippers")
        return keyframes
    
    def _parse_keyframes_from_xml(self, xml_path: str) -> Dict[str, Dict[str, any]]:
        """Parse keyframes from an XML file."""
        keyframes = {}
        
        try:
            xml_root = ET.parse(xml_path).getroot()
            keyframes_element = xml_root.find("keyframe")
            
            if keyframes_element is not None:
                for key in keyframes_element.findall("key"):
                    name = key.get("name", "unnamed")
                    qpos_str = key.get("qpos", "")
                    qpos = [float(x) for x in qpos_str.split()] if qpos_str else []
                    
                    keyframes[name] = {
                        "qpos": qpos,
                        "original_name": name
                    }
        except Exception as e:
            log.error(f"Failed to parse keyframes from {xml_path}: {e}")
        
        return keyframes
    
    def _collect_robot_keyframes(self) -> Dict[str, Dict[str, any]]:
        """Collect keyframes from robot XML file."""
        robot_config = self.config.get("robot", {})
        robot_path = robot_config.get("model_path", "")
        
        if not robot_path:
            return {}
        
        full_robot_path = self._get_full_path(robot_path)
        if not os.path.exists(full_robot_path):
            return {}
        
        keyframes = self._parse_keyframes_from_xml(full_robot_path)
        
        # Add robot-specific metadata
        for name, data in keyframes.items():
            data["source"] = "robot"
            log.info(f"Collected robot keyframe: {name}")
        
        return keyframes
    
    def _collect_gripper_keyframes(self) -> Dict[str, Dict[str, any]]:
        """Collect keyframes from gripper XML files."""
        keyframes = {}
        robot_config = self.config.get("robot", {})
        grippers = robot_config.get("grippers", {})
        
        if not grippers:
            return keyframes
        
        for gripper_name, gripper_config in grippers.items():
            gripper_path = gripper_config.get("model_path", "")
            gripper_type = gripper_config.get("type", "")
            # Extract position from gripper_name (left_gripper -> left, right_gripper -> right)
            position = gripper_name.split("_")[0] if "_" in gripper_name else gripper_name
            name_prefix = f"{position}_{gripper_type}_" if gripper_type else f"{position}_"
            
            if not gripper_path:
                continue
            
            full_gripper_path = self._get_full_path(gripper_path)
            if not os.path.exists(full_gripper_path):
                continue
            
            gripper_keyframes = self._parse_keyframes_from_xml(full_gripper_path)
            
            # Add gripper-specific metadata and rename keys
            for original_name, data in gripper_keyframes.items():
                unique_name = f"{gripper_name}_{original_name}"
                keyframes[unique_name] = {
                    **data,
                    "source": "gripper",
                    "gripper_name": gripper_name,
                    "name_prefix": name_prefix,
                    "position": position,
                    "type": gripper_type
                }
                log.debug(f"Collected gripper keyframe: {unique_name}")
        
        return keyframes
    
    def _save_scene_object_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Save current positions of scene objects."""
        scene_positions = {}
        scene_objects = self.config.get("scene_objects", [])
        
        for obj_config in scene_objects:
            obj_name = obj_config.get("name", "object")
            initial_pos = obj_config.get("initial_position", [0.0, 0.0, 0.0])
            scene_positions[obj_name] = (initial_pos[0], initial_pos[1], initial_pos[2])
                
        return scene_positions
    
    def _restore_scene_object_positions(self, scene_positions: Dict[str, Tuple[float, float, float]]) -> None:
        """Restore positions of scene objects."""
        for obj_name, (x, y, z) in scene_positions.items():
            try:
                body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                if body_id >= 0:
                    self.mj_data.xpos[body_id][0] = x
                    self.mj_data.xpos[body_id][1] = y
                    self.mj_data.xpos[body_id][2] = z
                    
                    # Also update qpos if the object has a free joint
                    joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj_name}_joint")
                    if joint_id >= 0:
                        qpos_addr = self.mj_model.jnt_qposadr[joint_id]
                        if qpos_addr + 2 < len(self.mj_data.qpos):
                            self.mj_data.qpos[qpos_addr] = x
                            self.mj_data.qpos[qpos_addr + 1] = y
                            self.mj_data.qpos[qpos_addr + 2] = z
            except:
                log.warning(f"Failed to restore position for scene object: {obj_name}")
    
    def _apply_dynamic_keyframes(self) -> None:
        """Apply dynamically collected keyframes to set initial mj_data state."""
        # Save scene object positions before applying keyframes
        scene_positions = self._save_scene_object_positions()
        
        # Collect keyframe data from XML files
        self.collected_keyframes = self._collect_keyframe_data()
        
        if not self.collected_keyframes:
            log.warning("No keyframes collected from robot and grippers")
            # Still restore scene object positions even if no robot keyframes
            self._restore_scene_object_positions(scene_positions)
            return
        
        # Apply composite keyframe to set initial state
        self._apply_composite_keyframe()
        
        # Restore scene object positions after keyframe application
        self._restore_scene_object_positions(scene_positions)
    
    def _apply_composite_keyframe(self, keyframe_name: str = None) -> None:
        """Apply a composite keyframe that combines robot and gripper states."""
        if not hasattr(self, 'collected_keyframes') or not self.collected_keyframes:
            log.warning("No collected keyframes available")
            return
        
        # Auto-detect suitable keyframe name if not provided
        if keyframe_name is None:
            keyframe_name = self._get_default_keyframe_name()
        
        # Initialize qpos array with zeros
        qpos_array = [0.0] * self.mj_model.nq
        
        # Step 1: Apply robot keyframe (includes robot body + left arm + right arm)
        if keyframe_name in self.collected_keyframes:
            robot_qpos = self.collected_keyframes[keyframe_name]["qpos"]
            
            # Robot keyframe should be applied to robot joints only, not gripper joints
            robot_joint_count = 0
            for joint_id in range(self.mj_model.njnt):
                joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                # Skip gripper joints (those with prefix)
                if joint_name and not self._is_gripper_joint(joint_name):
                    joint_qpos_addr = self.mj_model.jnt_qposadr[joint_id]
                    if robot_joint_count < len(robot_qpos) and joint_qpos_addr < len(qpos_array):
                        qpos_array[joint_qpos_addr] = robot_qpos[robot_joint_count]
                        robot_joint_count += 1
            
            log.info(f"Applied robot keyframe: {keyframe_name} to {robot_joint_count} robot joints")
        
        # Step 2: Apply gripper keyframes by finding joints with matching prefixes
        self._apply_gripper_keyframes_by_name(qpos_array)
        
        # Set the computed qpos to mj_data
        for i, qpos_val in enumerate(qpos_array):
            if i < self.mj_model.nq:
                self.mj_data.qpos[i] = qpos_val
        
        # Set controls to match joint positions for position-controlled joints
        if self.mj_model.nu > 0:
            joint_ids = self.mj_model.actuator_trnid[:, 0]
            qpos_addrs = self.mj_model.jnt_qposadr[joint_ids]
            valid_indices = qpos_addrs < len(self.mj_data.qpos)
            self.mj_data.ctrl[valid_indices] = self.mj_data.qpos[qpos_addrs[valid_indices]]
        
        log.info(f"Applied composite keyframe to {self.mj_model.nq} DOFs")
    
    def _get_default_keyframe_name(self) -> str:
        """Get the appropriate default keyframe name for the current robot."""
        if not hasattr(self, 'collected_keyframes') or not self.collected_keyframes:
            return "robot_home"
        
        # Priority order for common keyframe names
        priority_names = ["home", "robot_home", "default", "start", "initial"]
        
        for name in priority_names:
            if name in self.collected_keyframes:
                log.info(f"Using default keyframe: {name}")
                return name
        
        # If none of the priority names found, use the first available keyframe
        available_keyframes = list(self.collected_keyframes.keys())
        if available_keyframes:
            first_keyframe = available_keyframes[0]
            log.info(f"Using first available keyframe: {first_keyframe}")
            return first_keyframe
        
        # Fallback
        return "robot_home"
    
    def _is_gripper_joint(self, joint_name: str) -> bool:
        """Check if a joint belongs to a gripper based on name prefix, excluding robot arm joints."""
        robot_config = self.config.get("robot", {})
        grippers = robot_config.get("grippers", {})
        
        # First check if it's a robot arm joint (these should NOT be considered gripper joints)
        if joint_name.startswith("left_arm_") or joint_name.startswith("right_arm_") or \
           joint_name.startswith("body_") or joint_name.startswith("head_"):
            return False
        
        # Then check if it matches any gripper prefix
        for gripper_name, gripper_config in grippers.items():
            gripper_type = gripper_config.get("type", "")
            # Extract position from gripper_name (left_gripper -> left, right_gripper -> right)
            position = gripper_name.split("_")[0] if "_" in gripper_name else gripper_name
            name_prefix = f"{position}_{gripper_type}_" if gripper_type else f"{position}_"
            if name_prefix and joint_name.startswith(name_prefix):
                return True
        return False
    
    def _apply_gripper_keyframes_by_name(self, qpos_array: list) -> None:
        """Apply gripper keyframes by matching joint names with prefixes."""
        robot_config = self.config.get("robot", {})
        grippers = robot_config.get("grippers", {})
        
        for gripper_name, gripper_config in grippers.items():
            gripper_type = gripper_config.get("type", "")
            # Extract position from gripper_name (left_gripper -> left, right_gripper -> right)  
            position = gripper_name.split("_")[0] if "_" in gripper_name else gripper_name
            name_prefix = f"{position}_{gripper_type}_" if gripper_type else f"{position}_"
            
            # Find the corresponding gripper home keyframe
            gripper_keyframe_name = f"{gripper_name}_gripper_home"
            if gripper_keyframe_name not in self.collected_keyframes:
                log.warning(f"Gripper keyframe not found: {gripper_keyframe_name}")
                continue
            
            gripper_qpos = self.collected_keyframes[gripper_keyframe_name]["qpos"]
            log.info(f"Applying gripper keyframe: {gripper_keyframe_name} with prefix '{name_prefix}'")
            
            # Find joints that start with the gripper's name prefix
            gripper_joint_indices = []
            for joint_id in range(self.mj_model.njnt):
                joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name and joint_name.startswith(name_prefix):
                    joint_qpos_addr = self.mj_model.jnt_qposadr[joint_id]
                    gripper_joint_indices.append((joint_id, joint_qpos_addr, joint_name))
            
            # Sort by qpos address to maintain order
            gripper_joint_indices.sort(key=lambda x: x[1])
            
            # Apply gripper qpos to the corresponding joints
            applied_joints = 0
            for i, (joint_id, qpos_addr, joint_name) in enumerate(gripper_joint_indices):
                if i < len(gripper_qpos) and qpos_addr < len(qpos_array):
                    qpos_array[qpos_addr] = gripper_qpos[i]
                    applied_joints += 1
                    log.debug(f"  Joint {joint_name} (qpos[{qpos_addr}]) = {gripper_qpos[i]}")
            
            log.info(f"Applied {applied_joints}/{len(gripper_qpos)} values for {gripper_name}")
    
    def apply_keyframe_by_name(self, keyframe_name: str) -> bool:
        """Apply a specific collected keyframe by name."""
        if not hasattr(self, 'collected_keyframes') or keyframe_name not in self.collected_keyframes:
            log.error(f"Keyframe '{keyframe_name}' not found in collected keyframes")
            return False
        
        keyframe_data = self.collected_keyframes[keyframe_name]
        qpos = keyframe_data["qpos"]
        
        if len(qpos) <= self.mj_model.nq:
            # Apply the specific keyframe
            for i, qpos_val in enumerate(qpos):
                if i < self.mj_model.nq:
                    self.mj_data.qpos[i] = qpos_val
            
            # Update controls
            if self.mj_model.nu > 0:
                joint_ids = self.mj_model.actuator_trnid[:, 0]
                qpos_addrs = self.mj_model.jnt_qposadr[joint_ids]
                valid_indices = qpos_addrs < len(self.mj_data.qpos)
                self.mj_data.ctrl[valid_indices] = self.mj_data.qpos[qpos_addrs[valid_indices]]
            
            mujoco.mj_forward(self.mj_model, self.mj_data)
            log.info(f"Applied keyframe: {keyframe_name}")
            return True
        else:
            log.error(f"Keyframe size mismatch: {len(qpos)} values for {self.mj_model.nq} DOFs")
            return False
    
    def set_keyframe(self, keyframe_id: int) -> None:
        """Set the model to a specific keyframe."""
        if self.mj_model is None or self.mj_data is None:
            log.error("Model not initialized")
            return
        
        if keyframe_id < 0 or keyframe_id >= self.mj_model.nkey:
            log.error(f"Invalid keyframe ID: {keyframe_id}, available: 0-{self.mj_model.nkey-1}")
            return
        
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, keyframe_id)
        
        # Update controls to match the keyframe
        if self.mj_model.nu > 0:
            joint_ids = self.mj_model.actuator_trnid[:, 0]
            qpos_addrs = self.mj_model.jnt_qposadr[joint_ids]
            self.mj_data.ctrl[:] = self.mj_data.qpos[qpos_addrs]
        
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        keyframe_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, keyframe_id)
        log.info(f"Set to keyframe {keyframe_id}: {keyframe_name if keyframe_name else 'unnamed'}")
    
    def get_collected_keyframe_names(self) -> list:
        """Get list of collected keyframe names."""
        if not hasattr(self, 'collected_keyframes'):
            return []
        
        return list(self.collected_keyframes.keys())
    
    def get_keyframe_info(self) -> Dict[str, Dict[str, any]]:
        """Get detailed information about all collected keyframes."""
        if not hasattr(self, 'collected_keyframes'):
            return {}
        
        return self.collected_keyframes.copy()


def main():
    """Main function for testing the environment creator with command line config selection."""
    parser = argparse.ArgumentParser(description='MuJoCo Environment Creator')
    parser.add_argument('--config', '-c', 
                       type=str, 
                       default="simulation/scene_config/example.yaml",
                       help='Path to configuration YAML file (default: simulation/scene_config/example.yaml)')
    parser.add_argument('--output', '-o',
                       type=str,
                       default="generated_scene.xml", 
                       help='Output XML file path (default: generated_scene.xml)')
    parser.add_argument('--no-viewer', 
                       action='store_true',
                       help='Skip launching the MuJoCo viewer')
    
    args = parser.parse_args()
    
    try:
        env_creator = MujocoEnvCreator(config_path=args.config)
        
        print(f"Using configuration: {args.config}")
        print("Available robots:", [d.name for d in (env_creator.root_path / "assets").iterdir() 
                                  if d.is_dir() and not d.name.startswith("scene")])
        print("Available objects:", [d.name for d in (env_creator.root_path / "assets" / "scene_objects").iterdir() 
                                   if d.is_dir()])
        
        env_creator.save_xml(args.output)
        print(f"Generated XML saved to: {args.output}")
        
        try:
            model, data = env_creator.create_model()
            print(f"Successfully created model with {model.nq} DoFs, {model.nu} actuators")
            
            if not args.no_viewer:
                try:
                    viewer = mujoco.viewer.launch_passive(model, data)
                    while viewer.is_running():
                        mujoco.mj_step(model, data)
                        viewer.sync()
                except (ImportError, AttributeError):
                    print("MuJoCo viewer not available in this version")
            
        except Exception as e:
            print(f"Error creating model: {e}")
            
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Available configurations:")
        config_dir = Path(__file__).parent / "scene_config"
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                print(f"  {config_file.relative_to(Path(__file__).parent)}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()