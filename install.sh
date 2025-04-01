#!/bin/bash

echo "pip upgrade..."
pip3 install --upgrade pip

echo "packages installing..."
pip3 install openai==1.55.0
pip3 install google-ai-generativelanguage==0.6.10
pip3 install google-api-core==2.23.0
pip3 install google-api-python-client==2.154.0
pip3 install google-auth==2.36.0
pip3 install google-auth-httplib2==0.2.0
pip3 install google-generativeai==0.8.3
pip3 install googleapis-common-protos==1.66.0

echo "installation completed"