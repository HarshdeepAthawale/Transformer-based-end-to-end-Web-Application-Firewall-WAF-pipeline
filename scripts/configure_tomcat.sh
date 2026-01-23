#!/bin/bash
# Configure Tomcat for multiple applications on different ports

set -e

TOMCAT_DIR="/opt/tomcat9"
SERVER_XML="$TOMCAT_DIR/conf/server.xml"

if [ ! -f "$SERVER_XML" ]; then
    echo "Error: Tomcat server.xml not found at $SERVER_XML"
    exit 1
fi

echo "Configuring Tomcat for multiple applications..."

# Backup original
if [ ! -f "${SERVER_XML}.backup" ]; then
    cp "$SERVER_XML" "${SERVER_XML}.backup"
    echo "✓ Backed up original server.xml"
fi

# Create a Python script to properly modify the XML
python3 << 'PYTHON_SCRIPT'
import xml.etree.ElementTree as ET
import sys

server_xml = "/opt/tomcat9/conf/server.xml"

try:
    tree = ET.parse(server_xml)
    root = tree.getroot()
    
    # Find the Service element (usually named "Catalina")
    service = root.find(".//Service[@name='Catalina']")
    
    if service is not None:
        # Get the existing Connector and Engine
        connector = service.find(".//Connector[@port='8080']")
        engine = service.find(".//Engine[@name='Catalina']")
        
        if connector is not None and engine is not None:
            # Configure existing service for app1 on port 8080
            connector.set('port', '8080')
            connector.set('protocol', 'HTTP/1.1')
            connector.set('connectionTimeout', '20000')
            
            # Set appBase for app1
            host = engine.find(".//Host[@name='localhost']")
            if host is not None:
                host.set('appBase', 'webapps/app1')
            
            print("✓ Configured service for app1 on port 8080")
    
    # Add service for app2 on port 8081
    service2 = root.find(".//Service[@name='Catalina2']")
    if service2 is None:
        service2 = ET.SubElement(root, "Service", name="Catalina2")
        connector2 = ET.SubElement(service2, "Connector", 
                                   port="8081", 
                                   protocol="HTTP/1.1",
                                   connectionTimeout="20000")
        engine2 = ET.SubElement(service2, "Engine", name="Catalina2", defaultHost="localhost")
        host2 = ET.SubElement(engine2, "Host", name="localhost", appBase="webapps/app2")
        print("✓ Added service for app2 on port 8081")
    
    # Add service for app3 on port 8082
    service3 = root.find(".//Service[@name='Catalina3']")
    if service3 is None:
        service3 = ET.SubElement(root, "Service", name="Catalina3")
        connector3 = ET.SubElement(service3, "Connector", 
                                   port="8082", 
                                   protocol="HTTP/1.1",
                                   connectionTimeout="20000")
        engine3 = ET.SubElement(service3, "Engine", name="Catalina3", defaultHost="localhost")
        host3 = ET.SubElement(engine3, "Host", name="localhost", appBase="webapps/app3")
        print("✓ Added service for app3 on port 8082")
    
    # Write the modified XML
    tree.write(server_xml, encoding='utf-8', xml_declaration=True)
    print("✓ Tomcat configuration updated")
    
except Exception as e:
    print(f"Error configuring Tomcat: {e}")
    print("Using simpler configuration approach...")
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo "✓ Tomcat configuration complete"
else
    echo "Note: Using default Tomcat configuration. Applications will be accessible via context paths."
fi
