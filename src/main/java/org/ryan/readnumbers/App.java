package org.ryan.readnumbers;

import org.ryan.readnumbers.mnist.GetMnistImages;
import org.ryan.readnumbers.mnist.Image;

import java.util.List;

public class App {
    public static void main(String[] args) throws Exception {
        List<Image> trainingImages = GetMnistImages.getTrainingImages();
//        new App().test();

        for(Image ti : trainingImages) {
            System.out.println(ti);
        }

    }
}
