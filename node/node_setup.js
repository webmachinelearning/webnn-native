const webnn = require('./lib/webnn');
// navigator is undefined in node.js, but defined in electron.js.
if (global.navigator === undefined) {
  global.navigator = {};
}
global.navigator.ml = webnn.ml;
global.MLContext = webnn.MLContext
global.MLGraphBuilder = webnn.MLGraphBuilder
global.MLGraph = webnn.MLGraph
global.MLOperand = webnn.MLOperand
global.chai = require('chai');
global.fs = require('fs');