# Copyright 2021 The WebNN-native Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
import argparse
import os
import platform

def run():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Build ienn that is a c wrapper of nGraph.")
    parser.add_argument(
        '--webnn-native-lib-path',
        type=str,
        help='Copy ienn binary to the webnn native lib path.'
    )
    args = parser.parse_args()

    webnn_native_lib_path = args.webnn_native_lib_path

    sys = platform.system()
    if sys == "Windows":
        script_name = "build_ienn_msvc.bat"
    elif sys == "Linux":
        script_name = "build_ienn_unix.sh"
    output = subprocess.check_output(
        [os.path.join(os.path.abspath(os.path.dirname(__file__)), script_name),
        webnn_native_lib_path])
    if output.find(b"Build ienn succeeded") == -1:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(run())