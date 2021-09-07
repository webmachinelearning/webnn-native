const fse = require('fs-extra');

module.exports = {
  packagerConfig: {
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
      const node_path = '../../../';
      await fse.copy(`${node_path}node_setup.js`, 'node_setup.js', {overwrite: true});
      await fse.copy(`${node_path}lib`, 'lib', {overwrite: true});
      await fse.copy(`${node_path}build`, 'build', {overwrite: true});
      // Copy source code and dependences of semantic segmentation sample
      await fse.copy(`${node_path}third_party/webnn-samples/common`, 'common', {overwrite: true});
      await fse.copy(
        `${node_path}third_party/webnn-samples/test-data/models/deeplabv3_mnv2_nchw`,
        'test-data/models/deeplabv3_mnv2_nchw',
        {overwrite: true});
      await fse.copy(
        `${node_path}third_party/webnn-samples/test-data/models/deeplabv3_mnv2_nhwc`,
        'test-data/models/deeplabv3_mnv2_nhwc',
        {overwrite: true});
      await fse.copy(
        `${node_path}third_party/webnn-samples/semantic_segmentation`,
        'semantic_segmentation',
        {overwrite: true});

      console.log('Copy WebNN libs and samples successfully!');
    }
  }
}
