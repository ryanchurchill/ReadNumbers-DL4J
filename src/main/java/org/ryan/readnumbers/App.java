package org.ryan.readnumbers;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.ryan.readnumbers.mnist.GetMnistImages;
import org.ryan.readnumbers.mnist.Image;
import org.ryan.readnumbers.network.GetDataSetFromImages;

import java.util.List;

public class App {
    public static void main(String[] args) throws Exception {
        System.out.println("Initializing Data...");

        List<Image> trainingImages = GetMnistImages.getTrainingImages();
//        new App().test();

//        for(Image ti : trainingImages) {
//            System.out.println(ti);
//        }

        DataSet allData = new GetDataSetFromImages(trainingImages).getDataSet();

        // normalize
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);

        // mnist actually has a separate file with 10k test images, but we'll just split the 60k training images instead
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        System.out.println("Initializing Network...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(Image.PIXEL_LENGTH * Image.PIXEL_LENGTH) // Number of input datapoints.
                        .nOut(1000) // Number of output datapoints.
                        .activation("relu") // Activation function.
                        .weightInit(WeightInit.XAVIER) // Weight initialization.
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000)
                        .nOut(Image.NUM_LABELS)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        // create and train the network
        System.out.println("Training beginning...");
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.fit(trainingData);
        System.out.println("Training done");

        // test the network
        System.out.println("Testing beginning...");
        INDArray output = model.output(testData.getFeatureMatrix());
        Evaluation eval = new Evaluation(10);
        eval.eval(testData.getLabels(), output);

        System.out.println(eval.stats());
    }
}
