package xilicon;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class PreprocessedData {
    private static final int WINDOW_SIZE = 600;
    private static final int MIN_WORD_FREQUENCY = 3;

    private DataSetIterator trainIterator;
    private Map<String, Integer> tokenToIndexMap;

    public PreprocessedData(String filePath) throws IOException, InterruptedException {
        File inputFile = new File(filePath);
        List<String> tokens = tokenizeFile(inputFile);

        // Create a frequency map for the tokens
        Map<String, Integer> freqMap = createFrequencyMap(tokens);

        // Create a token-to-index map for tokens that occur at least MIN_WORD_FREQUENCY times
        tokenToIndexMap = createTokenToIndexMap(freqMap, MIN_WORD_FREQUENCY);

        // Convert the token sequence to an index sequence
        List<Integer> indexSequence = createIndexSequence(tokens, tokenToIndexMap);

        // Prepare the data for training using a sliding window approach
        List<DataSet> dataSets = createDataSets(indexSequence, WINDOW_SIZE, tokenToIndexMap.size());

        // Create a DataSetIterator
        trainIterator = new CustomDataSetIterator(dataSets);
    }
    private List<Integer> createIndexSequence(List<String> tokens, Map<String, Integer> tokenToIndexMap) {
        List<Integer> indexSequence = new ArrayList<>();
        for (String token : tokens) {
            if (tokenToIndexMap.containsKey(token)) {
                indexSequence.add(tokenToIndexMap.get(token));
            }
        }
        return indexSequence;
    }
    public Map<Integer, String> getIndexToTokenMap() {
        Map<Integer, String> indexToTokenMap = new HashMap<>();
        for (Map.Entry<String, Integer> entry : tokenToIndexMap.entrySet()) {
            indexToTokenMap.put(entry.getValue(), entry.getKey());
        }
        return indexToTokenMap;
    }
    private List<DataSet> createDataSets(List<Integer> indexSequence, int windowSize, int vocabSize) {
        List<DataSet> dataSets = new ArrayList<>();
        for (int i = 0; i < indexSequence.size() - windowSize; i++) {
            INDArray inputArray = Nd4j.zeros(new int[]{1, vocabSize, windowSize});
            INDArray outputArray = Nd4j.zeros(new int[]{1, vocabSize, windowSize});

            for (int j = 0; j < windowSize; j++) {
                int idx = indexSequence.get(i + j);
                inputArray.putScalar(new int[]{0, idx, j}, 1);
                if (j < windowSize - 1) {
                    outputArray.putScalar(new int[]{0, indexSequence.get(i + j + 1), j}, 1);
                }
            }

            DataSet dataSet = new DataSet(inputArray, outputArray);
            dataSets.add(dataSet);
        }

        return dataSets;
    }
    public DataSetIterator getTrainIterator() {
        return trainIterator;
    }

    public Map<String, Integer> getTokenToIndexMap() {
        return tokenToIndexMap;
    }
    public int getVocabSize() {
        return tokenToIndexMap.size();
    }
    // Additional helper methods
    // ...
    private Map<String, Integer> createFrequencyMap(List<String> tokens) {
        Map<String, Integer> freqMap = new HashMap<>();
        for (String token : tokens) {
            freqMap.put(token, freqMap.getOrDefault(token, 0) + 1);
        }
        return freqMap;
    }

    private Map<String, Integer> createTokenToIndexMap(Map<String, Integer> freqMap, int minWordFrequency) {
        Map<String, Integer> tokenToIndexMap = new HashMap<>();
        int index = 0;
        for (Map.Entry<String, Integer> entry : freqMap.entrySet()) {
            if (entry.getValue() >= minWordFrequency) {
                tokenToIndexMap.put(entry.getKey(), index++);
            }
        }
        return tokenToIndexMap;
    }

    private List<String> tokenizeFile(File inputFile) throws IOException {
        List<String> tokens = new ArrayList<>();
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(inputFile))) {
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] lineTokens = line.toLowerCase().trim().split("\\s+");
                Collections.addAll(tokens, lineTokens);
            }
        }
        return tokens;
    }
    public static class CustomDataSetIterator implements DataSetIterator {
        private List<DataSet> dataSets;
        private int position;

        public CustomDataSetIterator(List<DataSet> dataSets) {
            this.dataSets = dataSets;
            this.position = 0;
        }

        @Override
        public DataSet next(int num) {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public int inputColumns() {
            return dataSets.get(0).getFeatures().columns();
        }

        @Override
        public int totalOutcomes() {
            return dataSets.get(0).getLabels().columns();
        }

        @Override
        public boolean resetSupported() {
            return true;
        }

        @Override
        public boolean asyncSupported() {
            return false;
        }

        @Override
        public void reset() {
            position = 0;
        }

        @Override
        public int batch() {
            return 1;
        }

     //   @Override
        public int cursor() {
            return position;
        }

      //  @Override
        public int numExamples() {
            return dataSets.size();
        }

        @Override
        public void setPreProcessor(DataSetPreProcessor preProcessor) {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public DataSetPreProcessor getPreProcessor() {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public List<String> getLabels() {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public boolean hasNext() {
            return position < dataSets.size();
        }

        @Override
        public DataSet next() {
            return dataSets.get(position++);
        }
    }
}