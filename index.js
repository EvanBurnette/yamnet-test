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

// const noise_raw = await fetch("./noise.wav");
const noise_raw = await fetch("./noise.wav");
const audioCtx = new AudioContext();
const noise = await audioCtx.decodeAudioData(await noise_raw.arrayBuffer());
// console.log(noise);
const modelUrl = "https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1";
const model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });
const waveform = tf.tensor(noise.getChannelData(0));
const [scores, embeddings, spectrogram] = model.predict(waveform);
scores.print(true); // shape [N, 521]
embeddings.print(true); // shape [N, 1024]
// spectrogram.print(true); // shape [M, 64] // Find class with the top score when mean-aggregated across frames.
console.log(await scores.mean(0).argMax().array()); // Should print 494 corresponding to 'Silence' in YAMNet Class Map.

const predict_p = document.getElementById("predict");
predict_p.innerText = classes[await scores.mean(0).argMax().array()];

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

console.log(await tf.max(spectrogram).array());
// console.log(await spectrogram.array());

const spectrogram_scaled = await normalize(spectrogram)
  .NORMALIZED_VALUES.square()
  .square();

tf.browser.toPixels(spectrogram_scaled, canvas);
// console.log(await spectrogram_scaled.array());

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
