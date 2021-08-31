const fse = require('fs-extra');

module.exports = {
  packagerConfig: {
    prune: true,
    asar: {
      unpack: "node_setup.js",
      unpackDir: "{lib,build}"
   },
  },
  makers: [
    {
      name: '@electron-forge/maker-zip'
    }
   ],
  hooks: {
    generateAssets: async () => {
      // Copy WebNN libs
      await fse.copy('../../../node_setup.js', 'node_setup.js', {overwrite: true});
      await fse.copy('../../../lib', 'lib', {overwrite: true});
      await fse.copy('../../../build', 'build', {overwrite: true});
      // Copy source code and dependences of semantic segmentation sample
      await fse.copy('../../../third_party/webnn-samples/common', 'common', {overwrite: true});
      await fse.copy(
        '../../../third_party/webnn-samples/test-data/models/deeplabv3_mnv2_nchw',
        'test-data/models/deeplabv3_mnv2_nchw',
        {overwrite: true});
      await fse.copy(
        '../../../third_party/webnn-samples/test-data/models/deeplabv3_mnv2_nhwc',
        'test-data/models/deeplabv3_mnv2_nhwc',
        {overwrite: true});
      await fse.copy(
        '../../../third_party/webnn-samples/semantic_segmentation',
        'semantic_segmentation',
        {overwrite: true});

      console.log('Copy WebNN libs and samples successfully!');
    }
  }
}
