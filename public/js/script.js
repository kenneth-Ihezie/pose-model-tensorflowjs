const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1'
const EXAMPLE_IMG = document.getElementById('exampleImg')

let movenetModel;


async function loadAndRunModel(){
   movenetModel =  await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true })
   //let exampleInputTensor = tf.zeros([1, 192, 192, 3], 'int32')

   //convert images or videos to tensors
   let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG)
   console.log(imageTensor.shape);
    
   //[y, x, colorchannel]
   let cropStartPoint = [15, 170, 0]

   //[height, width, colorchannel(rgb)]
   let cropSize = [345, 345, 3]

   //perform the crop
   let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize)

   //tf.image.resizeBilinear(imageToBeResized, newShapeOfTheImage, alignCorners=true)
   let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt()
   console.log(resizedTensor.shape);


   //let tensorOutput = movenetModel.predict(exampleInputTensor)
   //let arrayOutput = await tensorOutput.array()
   
   //movenet expect a batch of dimension [1, 192, 192, 3] so we are using tf.expandDims to expnad the dimensions.
   let tensorOutput = movenetModel.predict(tf.expandDims(resizedTensor))
   let arrayOutput = await tensorOutput.array()
   console.log(arrayOutput);
}


loadAndRunModel()