#!/usr/bin/env python3
import argparse
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import time

try:
  from urllib2 import HTTPError, URLError, urlopen
except ImportError: # For Py3 compatibility
  from urllib.error import HTTPError, URLError
  from urllib.request import urlopen

import zipfile
import json


# The code is based on https://github.com/microsoft/DirectML/blob/master/Python/setup.py that uses
# the MIT license (https://github.com/microsoft/DirectML/blob/master/LICENSE). 

dml_feed_url = 'https://api.nuget.org/v3/index.json'
dml_resource_id = 'microsoft.ai.directml'
dml_resource_version = '1.5.1'

dependency_dir = '../../../../../third_party'
dml_bin_path = f'{dependency_dir}/{dml_resource_id}.{dml_resource_version}/bin/x64-win/'
base_path = os.path.dirname(os.path.realpath(__file__))
dependency_path = os.path.join(base_path, dependency_dir)

dml_resource_name = '.'.join([dml_resource_id, dml_resource_version])
dml_path = '%s\%s' % (dependency_path, dml_resource_name)

def get_resource_url(feed_url, resource_id, resource_version):
    index = urlopen(feed_url)
    resources = json.loads(index.read())['resources']

    for resource in resources:
        if resource['@type'] == 'PackageBaseAddress/3.0.0':
            return resource['@id'] + '/'.join([resource_id, resource_version]) + '/' + '.'.join([resource_id, resource_version]) + '.nupkg'

    return ''

def download_nupkg(feed_url, resource_id, resource_version, resource_path):
    if not os.path.exists(resource_path):
        url = get_resource_url(feed_url, resource_id, resource_version)
        if url:
            print('downloading ' + url)
            # download the package
            resource_file = resource_path + '.nupkg'
            with open(resource_file, 'wb') as file:
                result = urlopen(url)
                while True:
                  block = result.read(1024)
                  if not block:
                    break
                  file.write(block)

            if os.path.exists(resource_file):
                # nupkg is just a zip, unzip it
                with zipfile.ZipFile(resource_file, "r") as zip_ref:
                    zip_ref.extractall(resource_path)
                os.remove(resource_file)

def main():
    download_nupkg(dml_feed_url, dml_resource_id, dml_resource_version, dml_path)

if __name__ == '__main__':
  sys.exit(main())
