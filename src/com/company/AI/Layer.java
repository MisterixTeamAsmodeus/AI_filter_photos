package com.company.AI;


public class Layer {

    public int size; // размер слоя
    public double[] neurons; // нейроны слоя
    public double[] biases; // сдвиги нейрона
    public double[][] weights; // веса нейронов

    public Layer(int size, int nextSize) {
        this.size = size;
        neurons = new double[size];
        biases = new double[size];
        weights = new double[size][nextSize];
    }

}