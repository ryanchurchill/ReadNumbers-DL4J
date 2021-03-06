package org.ryan.readnumbers.mnist;

import javassist.bytecode.stackmap.TypeData;

import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class GetMnistImages {
    static final int EXPECTED_COUNT = 60000;
    static final String TRAINING_LABEL_FILE = "train-labels-idx1-ubyte";
    static final String TRAINING_IMAGE_FILE = "train-images-idx3-ubyte";

    public static List<Image> getTrainingImages() throws Exception
    {
        // STEP 1: build images from TRAINING_IMAGE_FILE
        List<Image> images = getImagesFromImageFile();

        // STEP 2: iterate through TRAINING_LABEL_FILE and add the labels to images
        setActualDigitsInImagesFromLabelFile(images);

        return images;
    }

    /**
     * Does not set actualDigits
     * @return
     * @throws Exception
     */
    private static List<Image> getImagesFromImageFile() throws Exception
    {
        List<Image> images = new ArrayList<>();

        InputStream imageStream = GetMnistImages.class.getResourceAsStream(TRAINING_IMAGE_FILE);
        // ignore first 4 bytes (magic number)
        byte[] imageStreamBuffer = new byte[4];
        imageStream.read(imageStreamBuffer);

        // next 4 bytes are the number of images
        imageStream.read(imageStreamBuffer);
        ByteBuffer wrapped = ByteBuffer.wrap(imageStreamBuffer);
        int actualSize = wrapped.getInt();
        if (actualSize != EXPECTED_COUNT) {
            throw new Exception("actual Size incorrect: " + Integer.toString(actualSize));
        }

        // next 4 bytes are row count
        imageStream.read(imageStreamBuffer);
        wrapped = ByteBuffer.wrap(imageStreamBuffer);
        int numberOfRows = wrapped.getInt();
        if (numberOfRows != Image.PIXEL_LENGTH) {
            throw new Exception("numberOfRows incorrect: " + Integer.toString(numberOfRows));
        }

        // next 4 bytes are column count
        imageStream.read(imageStreamBuffer);
        wrapped = ByteBuffer.wrap(imageStreamBuffer);
        int numberOfColumns = wrapped.getInt();
        if (numberOfColumns != Image.PIXEL_LENGTH) {
            throw new Exception("numberOfColumns incorrect: " + Integer.toString(numberOfColumns));
        }

        // now we parse through the rest. Each image is 784 bytes.
        imageStreamBuffer = new byte[784];

        while((imageStream.read(imageStreamBuffer)) != -1) {
            images.add(getImage(imageStreamBuffer));
        }

        imageStream.close();

        if (images.size() != EXPECTED_COUNT) {
            throw new Exception("images.size() = " + Integer.toString(images.size()));
        }

        return images;
    }

    /**
     * This assumes that order of images matches the order of the labels in the label file
     * @param images
     * @throws Exception
     */
    private static void setActualDigitsInImagesFromLabelFile(List<Image> images) throws Exception
    {
        InputStream labelStream = GetMnistImages.class.getResourceAsStream(TRAINING_LABEL_FILE);

        // 1. ignore first 4 bytes
        byte[] labelStreamBuffer = new byte[4];
        labelStream.read(labelStreamBuffer);

        // 2. next 4 bytes are the number of images
        labelStream.read(labelStreamBuffer);
        ByteBuffer wrapped = ByteBuffer.wrap(labelStreamBuffer);
        int actualSize = wrapped.getInt();
        if (actualSize != EXPECTED_COUNT) {
            throw new Exception("actual Size incorrect: " + Integer.toString(actualSize));
        }

        // now we parse through the rest. Each label is a single byte
        labelStreamBuffer = new byte[EXPECTED_COUNT];
        labelStream.read(labelStreamBuffer);
        int counter = 0;
        for (byte b : labelStreamBuffer) {
            images.get(counter).setActualDigit(b & 0xff);
            counter++;
        }

        // sanity check that we've reached the end of the buffer
        if (labelStream.read() != -1) {
            throw new Exception("labelStream longer than expected");
        }
        labelStream.close();
    }

    /**
     * Expect 784 bytes total: the pixels
     * TODO: this is not memory efficient!
     * @param bytes
     */
    private static Image getImage(byte[] bytes) throws Exception
    {
        // validation
        if (bytes.length != 784) {
            throw new Exception("bytes did not have expected length: " + Integer.toString(bytes.length));
        }
        return new Image(bytes);
    }
}
