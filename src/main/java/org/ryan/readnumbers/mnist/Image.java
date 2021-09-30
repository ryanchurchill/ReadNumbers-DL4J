package org.ryan.readnumbers.mnist;

/**
 * Not normalized
 *
 * we have an array of 128 x 128
 * which each have a value of 0-255
 * 0 means background (white), 255 means foreground (black).
 */
public class Image {
    public final static int PIXEL_LENGTH = 28;
    final static int ORIGINAL_MIN = 0;
    final static int ORIGINAL_MAX = 255;

    // X by Y (column by row)
    // from the file, we get a range of 0-255.
    int[][] pixels = new int[PIXEL_LENGTH][PIXEL_LENGTH];

    public void setActualDigit(int actualDigit) {
        this.actualDigit = actualDigit;
    }

    public int getActualDigit() {
        return actualDigit;
    }

    /* 0-9 AKA label - what digit does this image actually represent? */
    int actualDigit = -1;

    //

    /**
     * Create an Image from a byte array that stores the pixel values
     * Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
     * This does *not* set actualDigit - the label is typically set separately from a different file. (A less lazy
     * programmer might use dual iteration.)
     * @param bytes
     * @throws Exception
     */
    public Image(byte[] bytes) throws Exception
    {
        if (bytes.length != PIXEL_LENGTH * PIXEL_LENGTH) {
            throw new Exception("unexpected bytes.length: " + Integer.toString(bytes.length));
        }

//        if (_actualDigit < 0 || _actualDigit > 9) {
//            throw new Exception("Invalid digit: " + Integer.toString(_actualDigit));
//        }
//
//        this.actualDigit = _actualDigit;

        int xPos = 0;
        int yPos = 0;

        for (byte b : bytes)
        {
            int value = b & 0xff;
            pixels[xPos][yPos] = value;

            xPos++;
            if (xPos == PIXEL_LENGTH) {
                xPos = 0;
                yPos++;
            }
        }
    }

    /**
     * Original bytes of mnist file had 1d, row-wise list of pixels
     *
     * To encapsulate/visualize/draw an image, we stored this in a 2d (x,y) array
     *
     * Now we're giving a way to get the pixel at the 1D index (AKA the original index from the bytes), since this
     * is what we need for the network.
     *
     * TODO: I haven't tested if we really got the x and y right - it probably doesn't matter for the sake of the
     * network, but it's worth knowing.
     *
     * @param i
     * @return
     */
    public int getPixelAt1DIndex(int i)
    {
        int x = i % PIXEL_LENGTH; // remainder
        int y = i / PIXEL_LENGTH; // remainder is thrown away
        return pixels[x][y];
    }

    public String toString()
    {
        StringBuffer sb = new StringBuffer();
        sb.append("Actual: " + Integer.toString(actualDigit));
        sb.append(getDrawing());

        // what was this?
//        for (int y = 0; y < PIXEL_LENGTH; y++) {
//            sb.append("\n");
//            for (int x=0; x < PIXEL_LENGTH; x++) {
//                double num = pixels[x][y];
//                sb.append(String.format("%03d", num));
//                sb.append(' ');
//            }
//        }
        return sb.toString();
    }

    public String getDrawing()
    {
        StringBuffer sb = new StringBuffer();

        for (int y = 0; y < PIXEL_LENGTH; y++) {
            sb.append("\n");
            for (int x=0; x < PIXEL_LENGTH; x++) {
                double num = pixels[x][y];
                if (num > 100) {
                    sb.append('.');
                } else {
                    sb.append(' ');
                }
                sb.append(' '); // pad with a space to try and spread the image out
            }
        }
        return sb.toString();
    }
}
