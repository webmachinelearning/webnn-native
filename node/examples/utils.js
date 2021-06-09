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
const assert = require('assert').strict;

function almostEqual(
    a, b, episilon = 1e-6, rtol = 5.0 * 1.1920928955078125e-7) {
  const delta = Math.abs(a - b);
  if (delta <= episilon + rtol * Math.abs(b)) {
    return true;
  } else {
    console.warn(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

function checkValue(output, expected) {
  assert.ok(output.length === expected.length);
  for (let i = 0; i < output.length; ++i) {
    assert.ok(almostEqual(output[i], expected[i]));
  }
}

function sizeOfShape(array) {
  return array.reduce(
      (accumulator, currentValue) => accumulator * currentValue);
}

function checkShape(shape, expected) {
  assert.isTrue(shape.length === expected.length);
  for (let i = 0; i < shape.length; ++i) {
    assert.isTrue(shape[i] === expected[i]);
  }
}

module.exports = {
  checkValue
}
