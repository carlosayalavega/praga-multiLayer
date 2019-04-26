//carlos ayala vega a01337893
package com.company;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static double sigm(double x){
        return 1.0/(1.0  + Math.pow(Math.E,-x));
    }

    public static double sigmDer(double x){
        return sigm(x)*(1.0-sigm(x));
    }

    public static void main(String[] args){

        // Read the input

        List<String> lines = new ArrayList<>();

        try {
            lines = Files.readAllLines(Paths.get("train_data_ML.txt"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[] x1 = new double[lines.size()];
        double[] x2 = new double[lines.size()];
        double[] y = new double[lines.size()];

        for (int i = 0; i<lines.size();i++) {
            String [] parts = lines.get(i).split(" ");
            x1[i] = Double.parseDouble(parts[0]);
            x2[i] = Double.parseDouble(parts[1]);
            y[i] = Double.parseDouble(parts[2]);
        }

        int hidden = 80; // number of hidden neurons
        int samples = lines.size();

        double [] w1 = new double[hidden]; // weights between X1 and hidden neurons
        double [] w2 = new double[hidden]; // weights between X2 and hidden neurons
        double [] b = new double[hidden]; // bias of hidden neurons
             for (int i = 0; i < hidden ; i++) {
                    b[i] = Math.random()*2-1;
                    w1[i] = Math.random()*2-1;
                    w2[i] = Math.random()*2-1;
                }
        double [] p = new double[hidden]; // inner potential of hidden neurons
        double [] v = new double[hidden]; // output of hidden neurons
            
            for (int i = 0; i < hidden ; i++) {
                p[i] = Math.random()*2-1;
            }
            
        double B ; // bias of output neuron
        double [] inPot = new double[hidden];
        double [] Wo = new double[hidden]; // weights between hidden and output neuron
        double P, o ; // output and inner potential of output neuron
      
        for (int i = 0; i < hidden ; i++) {
            w1[i] = Math.random()*2-1;
            w2[i] = Math.random()*2-1;
            b[i] = Math.random()*2-1;
            Wo[i] = Math.random()*2-1;
        }

        B = Math.random()*2-1; // bias of output neuron
        double error, errorSum = 0;
        double LR = 0.03; 
        int epochs = 5000;

        for (int j = 0; j < epochs ; j++) {

            for (int i = 0; i < samples; i++) {

                // calculate output of hidden layer
                 for (int h=0; h<hidden; h++) {
                    inPot[h] = (x1[h]*w1[h])+(x2[h]*w2[h]);
                    Wo[h] = sigm (inPot[h]);
                 }
                // calculate output of output neuron
                P=0;

                for (int k=0; k < hidden; k++) {
                    P += b[k]*Wo[k];
                }
                // calculate error
                o =sigm(P);
                error = .5 * Math.pow((y[i] - o), 2);
                errorSum += error;
                // update weights
               
                for(int f =0; f<hidden; f++){
                    w1[f] += (y[i]-o)* sigmDer(P)*Wo[f]*sigmDer(inPot[f])*x1[i]*LR;
                    w2[f] += (y[i]-o)* sigmDer(P)*Wo[f]*sigmDer(inPot[f])*x2[i]*LR;
                    b[f] += (y[i]-o)* sigmDer(P)*Wo[f]*LR;
                }
            }
            System.out.println(errorSum/samples);
            errorSum = 0;
        }

    }
}
