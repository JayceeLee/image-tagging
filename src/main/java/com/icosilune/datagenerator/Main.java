/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.icosilune.datagenerator;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 *
 * @author ashmore
 */
public class Main {
  public static void main(String args[]) throws IOException {
   
    int toGenerate = 1000;
    String filePrefix = "output/";
    
    ImageGenerator imageGenerator = new ImageGenerator();

    File tagFile = new File(filePrefix+"tags.txt");
    try (FileWriter fileWriter = new FileWriter(tagFile)) {
      for(int i=0;i<toGenerate;i++) {
        ImageAndTags result = imageGenerator.generate();
        
        System.out.println("Writing "+i);
        String filename = String.format("out%03d.png", i);
        ImageIO.write(result.getImage(), "png", new File(filePrefix+filename));
        fileWriter.append(String.format("%s %s\n", filename, result.getTags()));
      }
    }
  }
}
