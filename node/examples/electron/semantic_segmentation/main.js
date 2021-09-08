// Modules to control application life and create native browser window
const {app, BrowserWindow} = require('electron')
const path = require('path')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow = {}

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1220,
    height: 840,
    webPreferences: {
      nodeIntegrationInWorker: true,
      contextIsolation: false,
      enableBlinkFeatures: "WebAssemblySimd,WebAssemblyThreads",
      preload: path.join(__dirname, 'node_setup.js')
    }
  })

  let url = `file://${__dirname}/semantic_segmentation/index.html`
  let numRunsParam, deviceParam
  for (let argv of process.argv) {
    if (argv.startsWith("numRuns=") && !numRunsParam) {
      // Load the index.html with 'numRuns' to run inference multiple times.
      numRunsParam = argv
      url = deviceParam ? `${url}&${argv}` : `${url}?${argv}`
    }
    if (argv.startsWith("device=") && !deviceParam) {
      // Load the index.html with 'device' to set preferred kind of device used.
      deviceParam = argv
      url = numRunsParam ? `${url}&${argv}` : `${url}?${argv}`
    }
  }

  mainWindow.loadURL(url)

  // Emitted when the window is closed.
  mainWindow.on('closed', function() {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function() {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') app.quit()
})

app.on(
    'activate',
    function() {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (mainWindow === null) createWindow()
    })

    // In this file you can include the rest of your app's specific main process
    // code. You can also put them in separate files and require them here.
