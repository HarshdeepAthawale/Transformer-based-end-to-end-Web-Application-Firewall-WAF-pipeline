#!/bin/bash
# Script to create 3 sample WAR applications for testing

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOMCAT_DIR="/opt/tomcat9"
APPS_DIR="$PROJECT_DIR/war_apps"

echo "Creating sample WAR applications..."

# Create temporary directory for building WARs
mkdir -p "$APPS_DIR"
cd "$APPS_DIR"

# Create App 1: Simple Hello World Servlet
echo "Creating App 1..."
mkdir -p app1/WEB-INF/classes
cat > app1/WEB-INF/web.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<web-app xmlns="http://xmlns.jcp.org/xml/ns/javaee"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://xmlns.jcp.org/xml/ns/javaee
         http://xmlns.jcp.org/xml/ns/javaee/web-app_4_0.xsd"
         version="4.0">
    <servlet>
        <servlet-name>HelloServlet</servlet-name>
        <servlet-class>HelloServlet</servlet-class>
    </servlet>
    <servlet-mapping>
        <servlet-name>HelloServlet</servlet-name>
        <url-pattern>/hello</url-pattern>
        <url-pattern>/</url-pattern>
    </servlet-mapping>
</web-app>
EOF

cat > app1/WEB-INF/classes/HelloServlet.java << 'EOF'
import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class HelloServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("text/html");
        PrintWriter out = response.getWriter();
        out.println("<html><body>");
        out.println("<h1>App 1 - Hello World</h1>");
        out.println("<p>Request URI: " + request.getRequestURI() + "</p>");
        out.println("<p>Query String: " + request.getQueryString() + "</p>");
        out.println("</body></html>");
    }
    
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("application/json");
        PrintWriter out = response.getWriter();
        out.println("{\"status\":\"success\",\"app\":\"app1\"}");
    }
}
EOF

# Create App 2: API Endpoint
echo "Creating App 2..."
mkdir -p app2/WEB-INF/classes
cat > app2/WEB-INF/web.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<web-app xmlns="http://xmlns.jcp.org/xml/ns/javaee"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://xmlns.jcp.org/xml/ns/javaee
         http://xmlns.jcp.org/xml/ns/javaee/web-app_4_0.xsd"
         version="4.0">
    <servlet>
        <servlet-name>ApiServlet</servlet-name>
        <servlet-class>ApiServlet</servlet-class>
    </servlet>
    <servlet-mapping>
        <servlet-name>ApiServlet</servlet-name>
        <url-pattern>/api/*</url-pattern>
        <url-pattern>/</url-pattern>
    </servlet-mapping>
</web-app>
EOF

cat > app2/WEB-INF/classes/ApiServlet.java << 'EOF'
import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class ApiServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("application/json");
        PrintWriter out = response.getWriter();
        String path = request.getPathInfo();
        out.println("{\"app\":\"app2\",\"path\":\"" + path + "\",\"method\":\"GET\"}");
    }
    
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("application/json");
        PrintWriter out = response.getWriter();
        out.println("{\"app\":\"app2\",\"method\":\"POST\",\"status\":\"received\"}");
    }
}
EOF

# Create App 3: Data Processing
echo "Creating App 3..."
mkdir -p app3/WEB-INF/classes
cat > app3/WEB-INF/web.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<web-app xmlns="http://xmlns.jcp.org/xml/ns/javaee"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://xmlns.jcp.org/xml/ns/javaee
         http://xmlns.jcp.org/xml/ns/javaee/web-app_4_0.xsd"
         version="4.0">
    <servlet>
        <servlet-name>DataServlet</servlet-name>
        <servlet-class>DataServlet</servlet-class>
    </servlet>
    <servlet-mapping>
        <servlet-name>DataServlet</servlet-name>
        <url-pattern>/data/*</url-pattern>
        <url-pattern>/</url-pattern>
    </servlet-mapping>
</web-app>
EOF

cat > app3/WEB-INF/classes/DataServlet.java << 'EOF'
import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class DataServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("application/json");
        PrintWriter out = response.getWriter();
        out.println("{\"app\":\"app3\",\"endpoint\":\"data\",\"method\":\"GET\"}");
    }
    
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("application/json");
        PrintWriter out = response.getWriter();
        out.println("{\"app\":\"app3\",\"endpoint\":\"data\",\"method\":\"POST\"}");
    }
}
EOF

# Compile Java files
echo "Compiling Java servlets..."
for app in app1 app2 app3; do
    if [ -f "$app/WEB-INF/classes"/*.java ]; then
        # Find servlet-api.jar (usually in Tomcat)
        SERVLET_API="$TOMCAT_DIR/lib/servlet-api.jar"
        if [ -f "$SERVLET_API" ]; then
            javac -cp "$SERVLET_API" "$app/WEB-INF/classes"/*.java -d "$app/WEB-INF/classes"
            echo "✓ Compiled $app"
        else
            echo "Warning: servlet-api.jar not found. Creating placeholder servlets..."
            # Create simple HTML pages as fallback
            mkdir -p "$app"
            echo "<html><body><h1>$app</h1><p>Application is running</p></body></html>" > "$app/index.html"
        fi
    fi
done

# Deploy to Tomcat
echo "Deploying applications to Tomcat..."
if [ -d "$TOMCAT_DIR/webapps" ]; then
    # Copy applications
    cp -r app1 "$TOMCAT_DIR/webapps/"
    cp -r app2 "$TOMCAT_DIR/webapps/"
    cp -r app3 "$TOMCAT_DIR/webapps/"
    echo "✓ Applications deployed to $TOMCAT_DIR/webapps/"
else
    echo "Warning: Tomcat webapps directory not found. Applications created in $APPS_DIR"
fi

echo "Done! Applications created."
