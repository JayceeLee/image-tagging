/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.icosilune.datagenerator;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import javax.imageio.ImageIO;

/**
 *
 * @author ashmore
 */
public class Main {
  public static void main(String args[]) throws IOException {
    
    int toGenerate = 1000;

    if(args.length > 0) {
      toGenerate = Integer.parseInt(args[0]);
    }
   
    String filePrefix = "output/";
    
    ImageGenerator imageGenerator = new ImageGenerator();
    
    Set<String> allTags = new HashSet<>();

    File tagFile = new File(filePrefix+"tags.txt");
    File tagMapFile = new File(filePrefix+"tag_map.txt");
    try (FileWriter tagWriter = new FileWriter(tagFile);
         FileWriter tagMapWriter = new FileWriter(tagMapFile)) {
      for(int i=0;i<toGenerate;i++) {
        ImageAndTags result = imageGenerator.generate();
        
        System.out.println("Writing "+i);
        String filename = String.format("out%04d.png", i);
        ImageIO.write(result.getImage(), "png", new File(filePrefix+filename));
        tagWriter.append(String.format("%s %s\n", filename, result.getTags()));
        
        for(String tag: result.getTags()) {
          if(!allTags.contains(tag)) {
            tagMapWriter.append(String.format("%s\n", tag));
            allTags.add(tag);
          }
        }
      }
    }
  }
}
