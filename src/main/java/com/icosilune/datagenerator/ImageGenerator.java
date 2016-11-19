/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.icosilune.datagenerator;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author ashmore
 */
public class ImageGenerator {
  
//  private static final int MIN_IMAGE_SIZE = 500;
//  private static final int MAX_IMAGE_SIZE = 500;
  private static final int IMAGE_SIZE = 500;
  
  private static final String[] FONT_NAMES = {
    "American Typewriter",
    "Arial",
    "Avenir",
    "Baskerville",
    "Chalkboard",
    "Comic Sans MS",
    "Damascus",
    "Geneva",
    "Impact",
  };
  
  private static final String[] GLYPHS = {
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
//    "H",
//    "I",
//    "J",
//    "K",
//    "L",
//    "M",
//    "N",
//    "O",
//    "P",
//    "Q",
//    "R",
//    "S",
//    "T",
//    "U",
//    "V",
//    "W",
//    "X",
//    "Y",
//    "Z",
  };
  
  private final Random rand = new Random();

  public ImageGenerator() {
  }
  
  public ImageAndTags generate() {
    
    
//    int width = rand.nextInt(MAX_IMAGE_SIZE-MIN_IMAGE_SIZE) + MIN_IMAGE_SIZE;
//    int height = rand.nextInt(MAX_IMAGE_SIZE-MIN_IMAGE_SIZE) + MIN_IMAGE_SIZE;
    int width = IMAGE_SIZE;
    int height = IMAGE_SIZE;
    
    BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    Graphics2D g = img.createGraphics();
    g.setColor(Color.WHITE);
    g.fillRect(0, 0, width, height);
    
    List<String> tags = new ArrayList<>();
    
    int glyphs = 1 + rand.nextInt(3);
    for(int i=0;i<glyphs;i++) {
      String glyph = GLYPHS[rand.nextInt(GLYPHS.length)];
      tags.add(glyph);
      drawGlyph(g, width, height, chooseFont(tags), glyph);
    }
    
    return ImageAndTags.create(img, tags);
  }
  
  Font chooseFont(List<String> tags) {
    String fontName = FONT_NAMES[rand.nextInt(FONT_NAMES.length)];
    tags.add(fontName.replace(" ", ""));
    int fontSize = 40 + rand.nextInt(40);
    return new Font(fontName, Font.PLAIN, fontSize);
  }
  
  void drawGlyph(Graphics2D g, int width, int height, Font font, String glyph) {
    // IIRC the coord system of graphics is upper left being the origin
    // NOTE: drawString draws with x and y being the lower left of the string.
    int x = 40 + rand.nextInt(width-80);
    int y = 40 + rand.nextInt(height-80);
    
    g.setColor(Color.BLACK);
    g.setFont(font);
    g.drawString(glyph, x, y);
  }
}
