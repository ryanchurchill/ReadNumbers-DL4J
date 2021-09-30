package org.ryan.readnumbers;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.ryan.readnumbers.mnist.GetMnistImages;
import org.ryan.readnumbers.mnist.Image;
import org.ryan.readnumbers.network.GetDataSetFromImages;

import java.util.List;

public class App {
    public static void main(String[] args) throws Exception {
        System.out.println("Initializing Data...");

        int rngSeed = 123;

        List<Image> trainingImages = GetMnistImages.getTrainingImages();
//        new App().test();

//        for(Image ti : trainingImages) {
//            System.out.println(ti);
//        }

        DataSet allData = new GetDataSetFromImages(trainingImages).getDataSet();

//        DataSetIterator mnistTrain = new MnistDataSetIterator(125, true, rngSeed);
//        DataSetIterator mnistTest = new MnistDataSetIterator(125, false, rngSeed);



        // normalize
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(allData);
        normalizer.transform(allData);

        // testing
//        Image image = trainingImages.get(1);
//        System.out.println(image);
//        System.out.println(allData.get(1));
        // end testing

        // mnist actually has a separate file with 10k test images, but we'll just split the 60k training images instead
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        System.out.println("Initializing Network...");
        int hiddenLayerSize = 100;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(new Nesterovs(0.006, 0.9))
//                .updater(new Adam())
                .regularization(true).l2(1e-4)
                .list()
                // hidden layer:
                .layer(0, new DenseLayer.Builder()
                        .nIn(Image.PIXEL_LENGTH * Image.PIXEL_LENGTH)
                        .nOut(hiddenLayerSize)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())

                // output layer:
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(hiddenLayerSize)
                        .nOut(Image.NUM_LABELS)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false)
                .backprop(true) // use backpropagation to adjust weights
                .build();

        // create and train the network
        System.out.println("Training beginning...");
        int numEpochs = 100;
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for( int i=0; i<numEpochs; i++ ){
            System.out.println("epoch " + i);
            model.fit(trainingData);
//            model.fit(mnistTrain);

            // test the network
            INDArray output = model.output(testData.getFeatureMatrix());
//            INDArray output = model.output(mnistTest.next().getFeatureMatrix());
            Evaluation eval = new Evaluation(10);
            eval.eval(testData.getLabels(), output);

            System.out.println(eval.stats());
        }
//        model.fit(trainingData);
//        System.out.println("Training done");
    }
}
