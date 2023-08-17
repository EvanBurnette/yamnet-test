// parse class map csv
const rawClassMapCSV = await fetch("yamnet_class_map.csv");
const rawClasses = await rawClassMapCSV.text();
const classes = rawClasses
  .split("\r\n")
  .slice(1)
  .map((line) =>
    line
      .split(",")
      .slice(2)
      .join(",")
      .replace(/(\/|"|\\)/g, "")
  );

// get sound file and get audio data
const noise_raw = await fetch("./noise.wav");
const audioCtx = new AudioContext();
const noise = await audioCtx.decodeAudioData(await noise_raw.arrayBuffer());

// get ML model
const modelUrl = "https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1";
const model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });
const waveform = tf.tensor(noise.getChannelData(0));
const [scores, embeddings, spectrogram] = model.predict(waveform);
// scores.print(true); // shape [N, 521]
// embeddings.print(true); // shape [N, 1024]
spectrogram.print(true); // shape [M, 64] // Find class with the top score when mean-aggregated across frames.
// console.log("class number: ", await scores.mean(0).argMax().array()); // Should print 494 corresponding to 'Silence' in YAMNet Class Map.

// get top 10 classes
const top10 = await tf.topk(scores.mean(0), 10, true).indices.array();
// console.log("scores mean topk k=10", top10);
// create one string from 10 classes
let top10classes = "";
for (const i of top10) {
  top10classes += "\n";
  top10classes += classes[i];
}
// add predictions to UI
const predict_p = document.getElementById("predict");
predict_p.innerText = top10classes;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

console.log("minimum value :", await tf.min(spectrogram).array());
console.log("maximum value :", await tf.max(spectrogram).array());

const spectrogram_scaled = await tf
  .transpose(await normalize(spectrogram).NORMALIZED_VALUES, [1, 0])
  .square()
  .reverse(0);

tf.browser.toPixels(spectrogram_scaled, canvas);

function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    // find the min val in the tensor
    const MIN_VALUES = min || tf.min(tensor, 0);
    // find the max val in the tensor
    const MAX_VALUES = max || tf.max(tensor, 0);
    // subtract the min val from every val in tensor
    // store in a new tensor
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    // calculate the range size of possible values
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    // calculate the adjusted vals divided by the range size as a new tensor
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });
  return result;
}
