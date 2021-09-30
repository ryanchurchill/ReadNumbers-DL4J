package org.ryan.readnumbers.network;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.ryan.readnumbers.mnist.Image;

import java.util.List;

/**
 * Data is *not* normalized
 */
public class GetDataSetFromImages {
    private List<Image> images;
    private int numPixels;
    private int numImages;
    private int numLabels; // assumption is that label is 0, 1, ..., 9

    public GetDataSetFromImages(List<Image> images, int _numLabels) {
        this.images = images;
        this.numPixels = Image.PIXEL_LENGTH * Image.PIXEL_LENGTH;
        this.numImages = images.size();
        this.numLabels = _numLabels;
    }

    public DataSet getDataSet()
    {
        /*
        imageData[i][j]
        i is the index of the image
        j is the index of the pixel
        pixels per image are flattened into a 1D array. imageData is 2D because we have one array per image.
         */
        float[][] imageData = new float[numImages][numPixels];

        for (int imageCounter = 0; imageCounter < numImages; imageCounter++) {
            Image image = images.get(imageCounter);
            for (int pixelCounter = 0; pixelCounter < numPixels; pixelCounter++) {
                imageData[imageCounter][pixelCounter] = image.getPixelAt1DIndex(pixelCounter);
            }
        }

        /*
        labelData[i][j]
        i is the index of the image
        j is the index of the label
        So for every image, we have an array of 10 labels. 9 of them will be set to 0, 1 will be set to 1.
        I based this off of MnistDataFetcher.fetch. Let's see if it works!
         */
        float[][] labelData = new float[numImages][numLabels];
        for (int imageCounter = 0; imageCounter < numImages; imageCounter++) {
            Image image = images.get(imageCounter);
            for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
                if (image.getActualDigit() == labelIndex) {
                    labelData[imageCounter][labelIndex] = 1;
                } else {
                    labelData[imageCounter][labelIndex] = 0;
                }
            }
        }

        INDArray indImages = Nd4j.create(imageData);
        INDArray indLabels = Nd4j.create(labelData);
        return new DataSet(indImages, indLabels);
    }
}
