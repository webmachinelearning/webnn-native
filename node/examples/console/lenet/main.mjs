// Copyright 2021 The WebNN-native Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

import webnn from '../../../node_setup.js';

(async function main() {
  let module = await import('../../../third_party/webnn-samples/lenet/lenet.js');
  const lenet = new module.LeNet('../../../third_party/webnn-polyfill/test-data/models/lenet_nchw/weights/lenet.bin');
  let start = Date.now();
  const outputOperand = await lenet.load();
  console.log(
      `loading elapsed time: ${(Date.now() - start).toFixed(2)} ms`);

  start = Date.now();
  await lenet.build(outputOperand);
  const compilationTime = Date.now() - start;
  console.log(`compilation elapsed time: ${compilationTime.toFixed(2)} ms`);

  start = Date.now();
  const input = new Float32Array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,5,4,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,47,119,210,164,119,116,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,99,233,250,253,252,250,246,72,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,224,253,254,253,254,253,230,72,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,105,240,253,226,253,254,250,136,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,201,251,213,253,254,248,41,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,86,201,251,254,254,249,48,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,35,207,253,254,252,164,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,7,92,139,249,254,254,250,112,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,43,206,245,251,254,254,254,248,43,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,80,239,253,254,254,254,254,251,143,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,111,239,250,250,250,253,252,168,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,74,90,91,98,234,251,170,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,7,134,250,214,34,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,245,253,208,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,2,0,0,0,0,0,4,137,251,250,117,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,168,91,4,1,0,1,3,34,233,253,245,40,1,0,0,0,0,0,0,0,0,0,0,0,0,0,4,156,247,210,69,39,7,54,93,201,252,250,138,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,60,195,248,248,218,180,235,249,253,251,205,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,133,111,247,249,250,253,254,253,226,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,12,115,118,164,245,247,229,57,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,4,6,6,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);
  const result = await lenet.predict(input);
  const inferenceTime = Date.now() - start.toFixed(2);
  console.log(`execution elapsed time: ${inferenceTime.toFixed(2)} ms`);
  const expected = [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
  let utils = await import('../../../third_party/webnn-polyfill/test/utils.js');
  utils.checkValue(result, expected);
  console.log("success for outputs buffer " + result);
})();
