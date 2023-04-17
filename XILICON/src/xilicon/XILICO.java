package xilicon;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.Map;
import java.util.Random;

    public class XILICO {
        public static void main(String[] args) throws IOException, InterruptedException {
            System.out.println("LOADING XILICO");
            String filePath = "train.txt";
            PreprocessedData preprocessedData = new PreprocessedData(filePath);

            // Get the preprocessed data and token-to-index map
            DataSetIterator trainIterator = preprocessedData.getTrainIterator();
            Map<String, Integer> tokenToIndexMap = preprocessedData.getTokenToIndexMap();
            int vocabSize = preprocessedData.getVocabSize();

            int hiddenLayerSize = 200;
            int inputSize = preprocessedData.getVocabSize();
            int outputSize = preprocessedData.getVocabSize();
            // Build the network configuration
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Adam(0.001))
                    .weightInit(WeightInit.XAVIER)
                    .l2(0.001)
                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    .gradientNormalizationThreshold(1.0)
                    .list()
                    .layer(0, new LSTM.Builder().nIn(inputSize).nOut(hiddenLayerSize).activation(Activation.TANH).build())
                    .layer(1, new LSTM.Builder().nIn(hiddenLayerSize).nOut(hiddenLayerSize).activation(Activation.TANH).build())
                    .layer(2, new LSTM.Builder().nIn(hiddenLayerSize).nOut(hiddenLayerSize).activation(Activation.TANH).build())
                    .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(hiddenLayerSize).nOut(outputSize).build())
                    .validateOutputLayerConfig(false)
                    .build();


            // Create and initialize the neural network
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(100));
            System.out.println(model.summary());

            // Train the neural network
            int numEpochs = 50;
            for (int epoch = 0; epoch < numEpochs; epoch++) {
                System.out.println(generateText(model, preprocessedData, 100).replace("  ", "\n"));

                System.out.println("Epoch: " + (epoch + 1));
                model.fit(trainIterator);
                trainIterator.reset();
            }

            // Save the trained model
            //MultiLayerNetwork model = MultiLayerNetwork.load(new File("saved-model.zip"), true);
            System.out.println(generateText(model, preprocessedData, 1000));
            System.out.println(generateText(model, preprocessedData, 1000));
            System.out.println(generateText(model, preprocessedData, 1000));
            System.out.println(generateText(model, preprocessedData, 1000));
            ModelSerializer.writeModel(model, "saved-model.zip", true);

        }
        public static String generateText(MultiLayerNetwork model, PreprocessedData preprocessedData, int numCharacters) {
            StringBuilder output = new StringBuilder();
            Map<String, Integer> tokenToIndexMap = preprocessedData.getTokenToIndexMap();
            Map<Integer, String> indexToTokenMap = preprocessedData.getIndexToTokenMap();
            int vocabSize = preprocessedData.getVocabSize();

            // Sample a random character from the vocabulary
            int startIndex = new Random().nextInt(vocabSize);
            INDArray input = Nd4j.zeros(1, vocabSize, 1);
            input.putScalar(new int[]{0, startIndex, 0}, 1);
            output.append(indexToTokenMap.get(startIndex));

            // Generate the text
            for (int i = 0; i < numCharacters; i++) {
                // Get the output probabilities for the next character
                INDArray rnnOutput = model.rnnTimeStep(input);
                INDArray outputProbabilities = rnnOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(rnnOutput.size(2) - 1));

                // Sample the next character based on the output probabilities
                int nextIndex = sampleFromDistribution(outputProbabilities);

                // Append the new character to the generated text
                String nextToken = indexToTokenMap.get(nextIndex);
                output.append(nextToken);
                output.append(" ");

                // Prepare the input for the next iteration
                input = Nd4j.zeros(1, vocabSize, 1);
                input.putScalar(new int[]{0, nextIndex, 0}, 1);
            }

            return output.toString();
        }

        private static int sampleFromDistribution(INDArray distribution) {
            double[] probabilities = distribution.toDoubleVector();
            double[] cumulativeProbabilities = new double[probabilities.length];
            cumulativeProbabilities[0] = probabilities[0];

            for (int i = 1; i < probabilities.length; i++) {
                cumulativeProbabilities[i] = cumulativeProbabilities[i - 1] + probabilities[i];
            }

            double randomValue = Math.random();
            for (int i = 0; i < cumulativeProbabilities.length; i++) {
                if (randomValue <= cumulativeProbabilities[i]) {
                    return i;
                }
            }

            return cumulativeProbabilities.length - 1;
        }
    }
