package com.company;

import com.company.AI.AIGetFiles;
import com.company.AI.Layer;
import com.company.AI.NeuralNetwork;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) throws IOException {
        NeuralNetwork neuralNetwork = new NeuralNetwork(0.01, NeuralNetwork.sigmoid, NeuralNetwork.dsigmoid, 3, 5, 3);
        String directory = "Илья";
        if (!readWeightsAndBiases(neuralNetwork, directory)){
            AIGetFiles[] learnFiles = getLearnFiles(directory);
            for (AIGetFiles learnFile : learnFiles) {
                System.out.println("Learn " + learnFile.input);
                learnAI(neuralNetwork, learnFile.input, learnFile.output, directory);
            }
        }
        saveWeightsAndBiases(neuralNetwork, directory);
        AIGetFiles[] workFiles = getWorkFiles(directory);
        for (AIGetFiles workFile : workFiles) {
            System.out.println("Work " + workFile.input);
            workAI(neuralNetwork, workFile.input, workFile.output, directory);
        }
    }

    private static boolean readWeightsAndBiases(NeuralNetwork neuralNetwork, String directory) {
        try {
            File file = new File("data/" + directory + "/Data/data.txt");
            FileReader fileReader = new FileReader(file);
            BufferedReader reader = new BufferedReader(fileReader);
            Layer[] layers = neuralNetwork.getLayers();
            for (Layer layer : layers) {
                for (int i = 0; i < layer.biases.length; i++) {
                    layer.biases[i] = Double.parseDouble(reader.readLine());
                }
                for (int i = 0; i < layer.weights.length; i++) {
                    for (int j = 0; j < layer.weights[i].length; j++) {
                        layer.weights[i][j] = Double.parseDouble(reader.readLine());
                    }
                }
            }
            reader.close();
            fileReader.close();
            neuralNetwork.setLayers(layers);
            return true;
        } catch (Exception e){
            return false;
        }
    }

    private static void saveWeightsAndBiases(NeuralNetwork neuralNetwork, String directory) throws IOException {
        Layer[] layers = neuralNetwork.getLayers();
        StringBuilder data = new StringBuilder();
        for (Layer layer : layers){
            for (int i = 0; i < layer.biases.length; i++){
                data.append(layer.biases[i]).append("\n");
            }
            for (int i = 0; i < layer.weights.length; i++){
                for (int j = 0; j < layer.weights[i].length; j++){
                    data.append(layer.weights[i][j]).append("\n");
                }
            }
        }
        if (!new File("data/" + directory + "/Data/").exists()){
            new File("data/" + directory + "/Data/").mkdir();
        }
        File file = new File("data/" + directory + "/Data/data.txt");
        file.createNewFile();
        FileWriter writer = new FileWriter(file, false);
        writer.write(data.toString());
        writer.close();
    }

    private static AIGetFiles[] getWorkFiles(String directory) {
        File learnDirectory = new File("data/" + directory + "/Image/");
        File[] learnFiles = learnDirectory.listFiles();
        Arrays.sort(learnFiles);
        ArrayList<AIGetFiles> aiGetFiles = new ArrayList<>();
        for (File file : learnFiles) {
            aiGetFiles.add(new AIGetFiles(file.getName(), file.getName() ));
        }
        return aiGetFiles.toArray(new AIGetFiles[0]);
    }

    private static AIGetFiles[] getLearnFiles(String directory) {
        File learnDirectory = new File("data/" + directory + "/Learn/");
        File[] learnFiles = learnDirectory.listFiles();
        Arrays.sort(learnFiles);
        ArrayList<AIGetFiles> aiGetFiles = new ArrayList<>();
        for (File file : learnFiles) {
            if (file.getName().contains("input")) {
                aiGetFiles.add(new AIGetFiles(file.getName(), file.getName().replace("input", "output")));
            } else {
                break;
            }
        }
        return aiGetFiles.toArray(new AIGetFiles[0]);
    }

    private static void workAI(NeuralNetwork neuralNetwork, String inputFiles, String outputFiles, String directory) throws IOException {
        BufferedImage inputImage = ImageIO.read(new File("data/" + directory + "/Image/" + inputFiles));
        WritableRaster inputRaster = inputImage.getRaster();
        for (int x = 0; x < inputRaster.getWidth(); x++) {
            for (int y = 0; y < inputRaster.getHeight(); y++) {
                double[] color = inputRaster.getPixel(x, y, new double[4]);
                double[] out = neuralNetwork.mathOutputFeedForward(new double[]{color[0] / 255, color[1] / 255, color[2] / 255});
                color[0] = 255 * out[0];
                color[1] = 255 * out[1];
                color[2] = 255 * out[2];
                inputRaster.setPixel(x, y, color);
            }
        }
        inputImage.setData(inputRaster);
        if (!new File("data/" + directory + "/Output/").exists()){
            new File("data/" + directory + "/Output/").mkdir();
        }
        ImageIO.write(inputImage, outputFiles.substring(outputFiles.lastIndexOf(".") + 1), new File("data/" + directory + "/Output/" + outputFiles));
    }

    private static void learnAI(NeuralNetwork neuralNetwork, String inputFilesLearn, String outputFilesLearn, String directory) throws IOException {
        File inputFile = new File("data/" + directory + "/Learn/" + inputFilesLearn);
        BufferedImage inputImage = ImageIO.read(inputFile);

        File outputFile = new File("data/" + directory + "/Learn/" + outputFilesLearn);
        BufferedImage outputImage = ImageIO.read(outputFile);

        WritableRaster inputRaster = inputImage.getRaster();
        WritableRaster outputRaster = outputImage.getRaster();

        for (int x = 0; x < inputRaster.getWidth(); x++) {
            for (int y = 0; y < inputRaster.getHeight(); y++) {
                double[] inputPixel = inputRaster.getPixel(x, y, new double[4]);
                double[] outputPixel = outputRaster.getPixel(x, y, new double[4]);
                if (outputPixel[1] == 0 && outputPixel[2] == 0) {
                    outputPixel[1] = outputPixel[0];
                    outputPixel[2] = outputPixel[0];
                }
                neuralNetwork.mathOutputFeedForward(new double[]{inputPixel[0] / 255, inputPixel[1] / 255, inputPixel[2] / 255});
                neuralNetwork.backpropagation(new double[]{outputPixel[0] / 255, outputPixel[1] / 255, outputPixel[2] / 255});
            }
        }
    }
}