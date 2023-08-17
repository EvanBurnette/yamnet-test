# Yamnet Test

This is a itsy bitsy demo app to test YAMNET in browser with Tensorflow js. [Yamnet](https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1) is an audio classifier model from the tensorflow hub which is based on MobileNet V1.

## Usage

1. `git clone this repo`
1. `npm install` or `pnpm install`
1. `npm vite` or `pnpm vite`

## Why?

This is to test creating spectrograms and auto sorting features for an online sample pack creator I'm working on. I want to try using the embeddings to do K-nearest neighbor analysis to find the most different sounds in a user's uploaded recordings. The thinking here is that the most different sounds could be the most special, but it remains to be seen.

I would also like to explore using earlier convolution steps to categorize the personality or adjective tag of the sample and use the final embedding to detect the broader class, i.e. spicy kick, sassy snare, sharp high-hat.
