const webnn = require('../../lib/webnn');
global.navigator.ml = webnn.ml;
global.MLGraphBuilder = webnn.MLGraphBuilder;